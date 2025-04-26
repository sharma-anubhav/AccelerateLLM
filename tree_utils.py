import torch
import time

class TrieNode:
    def __init__(self, token_id=None, parent=None):
        self.token_id = token_id
        self.parent = parent
        self.children = {}
        self.is_end_of_beam = False

    def add_child(self, token_id):
        if token_id not in self.children:
            self.children[token_id] = TrieNode(token_id, parent=self)
        return self.children[token_id]

    def get_sequence(self):
        seq = []
        curr = self
        while curr.parent is not None:
            seq.append(curr.token_id)
            curr = curr.parent
        return list(reversed(seq))

def run_ssm_generation(tok, mdl, input_ids_list, num_beams, max_depth):
    input_ids = torch.tensor(input_ids_list, device=mdl.device).unsqueeze(0)
    output_ids = mdl.generate(
        input_ids=input_ids,
        max_new_tokens=max_depth,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tok.pad_token_id,
    )

    generated_sequences = []
    input_len = len(input_ids_list)
    for ids in output_ids:
        padding_indices = (ids[input_len:] == tok.pad_token_id).nonzero()
        if len(padding_indices) > 0:
            first_pad_index = padding_indices[0].item()
            seq = ids[input_len: input_len + first_pad_index].cpu().tolist()
        else:
            seq = ids[input_len:].cpu().tolist()
        if seq:
            generated_sequences.append(seq)
    return generated_sequences

def normalize_and_dedupe(seqs, eos_id, depth):
    seen = set()
    out = []
    for seq in seqs:
        if len(seq) > depth:
            seq = seq[:depth]
        tup = tuple(seq)
        if tup in seen:
            continue
        seen.add(tup)
        out.append(seq)
    return out

def build_token_tree(seqs, eos_id):
    root = TrieNode(token_id=None, parent=None)
    if not seqs:
        return root
    for seq in seqs:
        node = root
        for tid in seq:
            node = node.add_child(tid)
        node.is_end_of_beam = True
    return root

def flatten_tree(root):
    nodes_list = []
    node_map = {}
    pos_map = {}
    def dfs(node):
        sorted_children = sorted(node.children.items())
        for token_id, child_node in sorted_children:
            node_map[child_node] = len(nodes_list)
            pos_map[child_node] = child_node
            nodes_list.append(token_id)
            dfs(child_node)
    dfs(root)
    return nodes_list, node_map, pos_map

def build_global_tree_mask(root, current_len, device):
    tree_tokens, node_to_idx_map, _ = flatten_tree(root)
    N = len(tree_tokens)
    total_len = current_len + N
    mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=torch.float)
    causal_mask_prefix = torch.tril(torch.zeros(current_len, current_len, device=device))
    mask[:current_len, :current_len] = causal_mask_prefix
    for u, u_idx in node_to_idx_map.items():
        global_pos_u = current_len + u_idx
        mask[global_pos_u, :current_len] = 0.0
        curr = u
        while curr is not None and curr.parent is not None:
            if curr in node_to_idx_map:
                ancestor_idx = node_to_idx_map[curr]
                global_pos_ancestor = current_len + ancestor_idx
                mask[global_pos_u, global_pos_ancestor] = 0.0
            curr = curr.parent
    return tree_tokens, mask, node_to_idx_map

def tree_decode_single_kernel(current_ids, root, target, target_tok):
    tree_tokens, attention_mask, node_to_idx_map = build_global_tree_mask(root, len(current_ids), target.device)
    if not tree_tokens:
        return {}, None, [], {}
    global_ids = current_ids + tree_tokens
    input_ids = torch.tensor([global_ids], device=target.device)
    final_mask = attention_mask.unsqueeze(0)
    with torch.no_grad():
        out = target(input_ids=input_ids, attention_mask=final_mask, use_cache=False, return_dict=True)
    logits = out.logits[0]
    preds = {}
    pos_predicting_first_token = len(current_ids) - 1
    first_t = logits[pos_predicting_first_token].argmax(-1).item()
    for node, node_idx in node_to_idx_map.items():
        logit_pos = len(current_ids) + node_idx
        if logit_pos < logits.shape[0]:
            predicted_next_token_id = logits[logit_pos].argmax(-1).item()
            preds[node] = predicted_next_token_id
    return preds, first_t, tree_tokens, node_to_idx_map

def verify_with_tree_kernel(current_ids, root, target, target_tok):
    eos = target_tok.eos_token_id
    if not root.children:
        inp = torch.tensor([current_ids], device=target.device)
        out = target.generate(input_ids=inp, max_new_tokens=1, do_sample=False, pad_token_id=eos)
        new_token_id = out[0, -1].item()
        return [new_token_id], current_ids + [new_token_id]
    preds, first_t, tree_tokens, node_to_idx_map = tree_decode_single_kernel(current_ids, root, target, target_tok)
    if first_t is None or first_t not in root.children:
        inp = torch.tensor([current_ids], device=target.device)
        out = target.generate(input_ids=inp, max_new_tokens=1, do_sample=False, pad_token_id=eos)
        new_token_id = out[0, -1].item()
        return [new_token_id], current_ids + [new_token_id]
    accepted_tokens = []
    current_node = root.children[first_t]
    accepted_tokens.append(current_node.token_id)
    while True:
        if current_node.token_id == eos:
            break
        target_next_token = preds.get(current_node)
        if target_next_token is None:
            break
        if target_next_token in current_node.children:
            current_node = current_node.children[target_next_token]
            accepted_tokens.append(current_node.token_id)
        else:
            accepted_tokens.append(target_next_token)
            break
    return accepted_tokens, current_ids + accepted_tokens
