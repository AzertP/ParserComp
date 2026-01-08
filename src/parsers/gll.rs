use std::collections::{HashMap, HashSet, VecDeque};
use std::{u32, vec};

use crate::grammars::{NumSymbol, NumericGrammar};
use crate::parse_tree::{ParseSymbol, ParseTree};

type GIndex = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GKind {
    Terminal(u32),
    NonTerminal(u32),
    Epsilon,
    LHS(u32),
    Alt,
    End,
    EOS,
}

#[derive(Debug)]
struct GNode {
    kind: GKind,
    seq: GIndex,
    alt: Option<GIndex>,
}
struct GrammarGraph {
    nodes: Vec<GNode>,
    headers: HashMap<u32, GIndex>,
    eos_index: GIndex,
}

impl GrammarGraph {
    /// Convert grammar to the presentation in the paper
    pub fn from_numeric(grammar: &NumericGrammar) -> Self {
        let mut nodes = Vec::new();
        let mut headers = HashMap::new();

        for (&nt_id, _) in &grammar.rules {
            let idx = nodes.len();
            headers.insert(nt_id, idx);

            nodes.push(GNode {
                kind: GKind::LHS(nt_id),
                seq: 0, // Will point to first ALT
                alt: None,
            });
        }

        let nt_ids: Vec<u32> = grammar.rules.keys().cloned().collect();
        for nt_id in nt_ids {
            let lhs_idx = *headers.get(&nt_id).unwrap();
            let productions = &grammar.rules[&nt_id];

            let mut previous_alt_idx: Option<usize> = None;

            for (prod_idx, production) in productions.iter().enumerate() {
                let alt_idx = nodes.len();

                // Link LHS to the first ALT
                if prod_idx == 0 {
                    nodes[lhs_idx].seq = alt_idx;
                }

                // Link previous ALT to this ALT [cite: 114]
                if let Some(prev) = previous_alt_idx {
                    nodes[prev].alt = Some(alt_idx);
                }
                previous_alt_idx = Some(alt_idx);

                // Push the ALT node
                nodes.push(GNode {
                    kind: GKind::Alt,
                    seq: 0,
                    alt: None,
                });

                let mut current_seq_idx = alt_idx;

                if production.is_empty() {
                    let eps_idx = nodes.len();
                    nodes[current_seq_idx].seq = eps_idx;
                    nodes.push(GNode {
                        kind: GKind::Epsilon,
                        seq: 0,
                        alt: None,
                    });
                    current_seq_idx = eps_idx;
                } else {
                    for sym in production {
                        let sym_idx = nodes.len();
                        nodes[current_seq_idx].seq = sym_idx;

                        let kind = match sym {
                            NumSymbol::Terminal(id) => GKind::Terminal(*id),
                            NumSymbol::NonTerminal(id) => GKind::NonTerminal(*id),
                        };

                        nodes.push(GNode {
                            kind,
                            seq: 0,
                            alt: None,
                        });
                        current_seq_idx = sym_idx;
                    }
                }

                // 4. Add END node
                let end_idx = nodes.len();
                nodes[current_seq_idx].seq = end_idx;

                nodes.push(GNode {
                    kind: GKind::End,
                    seq: alt_idx,
                    alt: Some(lhs_idx),
                });
            }
        }

        // Finally we add the EOS node
        let eos_index = nodes.len();

        nodes.push(GNode {
            kind: GKind::EOS,
            seq: 0,
            alt: None,
        });
        GrammarGraph {
            nodes,
            headers,
            eos_index,
        }
    }

    pub fn get(&self, index: GIndex) -> &GNode {
        &self.nodes[index]
    }

    #[allow(dead_code)]
    pub fn get_mut(&mut self, index: GIndex) -> &mut GNode {
        &mut self.nodes[index]
    }
}

// ----------------------------------
// Graph Structured Stack
// ----------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Descriptor {
    gn: GIndex,
    i: u32,
    /// GSS node index
    sn: GSSNodeId,
    /// SPPF node index
    dn: Option<SPPFNodeId>,
}

// Graph Structured Stack
type GSSNodeId = usize;

// GSS is identified by (GrammarSlot, i)
struct GSSNode {
    gn: GIndex,
    i: u32,
    /// List of edges
    edges: HashSet<GSSEdge>,
    pops: HashSet<SPPFNodeId>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct GSSEdge {
    dst: GSSNodeId,
    sppf_node: Option<SPPFNodeId>,
}

#[derive(PartialEq, Eq, Hash)]
struct GSSNodeKey {
    gn: GIndex,
    i: u32,
}

struct GSS {
    nodes: Vec<GSSNode>,
    lookup: HashMap<GSSNodeKey, GSSNodeId>,
}

impl GSS {
    fn new() -> Self {
        GSS {
            nodes: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    fn add_node(&mut self, node: GSSNode) -> GSSNodeId {
        let new_id = self.nodes.len();
        self.nodes.push(node);
        new_id
    }

    pub fn get(&self, id: GSSNodeId) -> &GSSNode {
        &self.nodes[id]
    }

    pub fn get_mut(&mut self, id: GSSNodeId) -> &mut GSSNode {
        &mut self.nodes[id]
    }

    // Find or create
    pub fn find(&mut self, gn: GIndex, i: u32) -> GSSNodeId {
        let key = GSSNodeKey { gn, i };
        *self.lookup.entry(key).or_insert_with(|| {
            let new_id = self.nodes.len();
            self.nodes.push(GSSNode {
                gn,
                i,
                edges: HashSet::new(),
                pops: HashSet::new(),
            });
            new_id
        })
    }
}

// --------------------------------
// Shared Packed Parse Forest (SPPF)
// --------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SPPFNodeId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PackID(pub usize);

struct SPPFNode {
    gn: GIndex,
    li: u32,
    ri: u32,
    pack_ns: HashSet<PackID>,
}

struct SPPFPackingNode {
    gn: GIndex,
    pivot: u32,
    left_c: Option<SPPFNodeId>,
    right_c: SPPFNodeId,
}

struct SPPF {
    nodes: Vec<SPPFNode>,
    pack_nodes: Vec<SPPFPackingNode>,
    lookup_nodes: HashMap<(GIndex, u32, u32), SPPFNodeId>,
    lookup_packs: HashMap<(SPPFNodeId, GIndex, u32), PackID>,
}

impl SPPF {
    fn new() -> Self {
        SPPF {
            nodes: Vec::new(),
            pack_nodes: Vec::new(),
            lookup_nodes: HashMap::new(),
            lookup_packs: HashMap::new(),
        }
    }

    /// Find or create SPPF node
    fn find(&mut self, gn: GIndex, li: u32, ri: u32) -> SPPFNodeId {
        let key = (gn, li, ri);
        *self.lookup_nodes.entry(key).or_insert_with(|| {
            let new_id = SPPFNodeId(self.nodes.len());
            self.nodes.push(SPPFNode {
                gn,
                li,
                ri,
                pack_ns: HashSet::new(),
            });
            new_id
        })
    }

    /// Find or create Pack node
    fn find_pack(
        &mut self,
        parent: SPPFNodeId,
        gn: GIndex,
        pivot: u32,
        left_c: Option<SPPFNodeId>,
        right_c: SPPFNodeId,
    ) -> PackID {
        let key = (parent, gn, pivot);
        *self.lookup_packs.entry(key).or_insert_with(|| {
            let new_id = PackID(self.pack_nodes.len());
            self.pack_nodes.push(SPPFPackingNode {
                gn,
                pivot,
                left_c,
                right_c,
            });
            new_id
        })
    }

    fn get(&self, id: SPPFNodeId) -> &SPPFNode {
        &self.nodes[id.0]
    }

    fn get_mut(&mut self, id: SPPFNodeId) -> &mut SPPFNode {
        &mut self.nodes[id.0]
    }
}

// ----------------------------------
// GLL Parser
// --------------------------------

pub struct GLLParser {
    grammar: NumericGrammar,
    accepting: bool,
    input: Vec<u32>,

    desc_set: HashSet<Descriptor>,
    desc_queue: VecDeque<Descriptor>,

    /// Input position
    i: u32,

    g_grammar: GrammarGraph,
    gss: GSS,
    gss_root: GSSNodeId,
    sppf: SPPF,
    // Global context
    gn: GIndex,
    /// GSS node index
    sn: GSSNodeId,
    /// SPPF node index
    dn: Option<SPPFNodeId>,
}

impl GLLParser {
    pub fn new(grammar: &NumericGrammar) -> Self {
        GLLParser {
            grammar: grammar.clone(),
            gn: 0,
            sn: 0,
            dn: None,
            desc_set: HashSet::new(),
            desc_queue: VecDeque::new(),
            gss: GSS::new(),
            gss_root: 0,
            sppf: SPPF::new(),
            i: 0,
            g_grammar: GrammarGraph::from_numeric(grammar),
            accepting: false,
            input: Vec::new(),
        }
    }

    fn initialisation(&mut self) {
        self.gn = 0;
        self.sn = 0;
        self.dn = None;
        self.desc_set.clear();
        self.desc_queue.clear();
        self.gss = GSS::new();
        self.sppf = SPPF::new();
        self.i = 0;

        // Create gss root
        let start_gn = GSSNode {
            gn: self.g_grammar.eos_index,
            i: 0,
            edges: HashSet::new(),
            pops: HashSet::new(),
        };
        self.gss_root = self.gss.add_node(start_gn);

        self.sn = self.gss_root;

        // Traverse all rules from start symbol in grammar graph
        let start_nt_id = self.g_grammar.headers[&self.grammar.start];
        let mut current_alt = self.g_grammar.get(start_nt_id).seq;
        while current_alt != 0 {
            let start_gn = self.g_grammar.get(current_alt).seq;
            self.queue_descriptor(start_gn, 0, self.sn, None);

            let next_alt = self.g_grammar.get(current_alt).alt;
            match next_alt {
                Some(alt) => {
                    current_alt = alt;
                }
                None => break,
            }
        }
    }

    /// Add new descriptor to the queue, skip if already exists
    fn queue_descriptor(
        &mut self,
        gn: GIndex,
        i: u32,
        gss_n: GSSNodeId,
        sppf_n: Option<SPPFNodeId>,
    ) {
        let desc = Descriptor {
            gn,
            i,
            sn: gss_n,
            dn: sppf_n,
        };
        if !self.desc_set.contains(&desc) {
            self.desc_set.insert(desc.clone());
            self.desc_queue.push_back(desc);
        }
    }

    /// Get the top descriptor and unloads its fields into the global context variables.
    /// Returns false if the queue is empty.
    fn dequeue_descriptor(&mut self) -> bool {
        let temp = self.desc_queue.pop_front();
        match temp {
            Some(desc) => {
                // Set global context
                self.gn = desc.gn;
                self.i = desc.i;
                self.sn = desc.sn;
                self.dn = desc.dn;
                true
            }
            None => false,
        }
    }

    fn call(&mut self, gn: GIndex) {
        let return_gn = self.g_grammar.get(gn).seq;
        let gss_n: GSSNodeId = self.gss.find(return_gn, self.i);
        let gss_e = GSSEdge {
            dst: self.sn,
            sppf_node: self.dn,
        };

        let pops_to_process: Option<Vec<SPPFNodeId>> = {
            let node = self.gss.get_mut(gss_n);
            if !node.edges.contains(&gss_e) {
                node.edges.insert(gss_e);
                // Clone to vector is faster, probably
                Some(node.pops.iter().cloned().collect())
            } else {
                None
            }
        };

        if let Some(pops) = pops_to_process {
            for sppf_node in pops {
                let right_extent = self.sppf.get(sppf_node).ri;
                let new_dn = self.sppf_update(return_gn, self.dn, sppf_node);

                self.queue_descriptor(return_gn, right_extent, self.sn, Some(new_dn));
            }
        }

        let nt_id = match self.g_grammar.get(gn).kind {
            GKind::NonTerminal(id) => id,
            _ => panic!("call() invoked on non-NonTerminal node"),
        };
        let lhs_idx = *self
            .g_grammar
            .headers
            .get(&nt_id)
            .expect("NonTerminal not found in headers");
        let mut current_alt = self.g_grammar.get(lhs_idx).seq;

        // Loop through all alternatives (productions) for this Non-Terminal
        // Paper: "for (GNode p = prules(gn).alt; p != null; p = p.alt)"
        while current_alt != 0 {
            // Assuming 0 is null/invalid in your GIndex
            let (production_start, next_alt) = {
                let alt_node = self.g_grammar.get(current_alt);
                (alt_node.seq, alt_node.alt)
            };
            self.queue_descriptor(production_start, self.i, gss_n, None);

            if let Some(next) = next_alt {
                current_alt = next;
            } else {
                break;
            }
        }
    }

    fn sppf_update(&mut self, gn: GIndex, ln: Option<SPPFNodeId>, rn: SPPFNodeId) -> SPPFNodeId {
        // ret = self.find(gn, li, ri)
        let left = match ln {
            Some(id) => self.sppf.get(id).li,
            None => self.sppf.get(rn).li,
        };

        let right = self.sppf.get(rn).ri;

        let current_gnode = self.g_grammar.get(gn);
        let next_gn;
        // Check if END node
        if current_gnode.kind == GKind::End {
            next_gn = current_gnode
                .alt
                .expect("END node must have alt pointing to LHS");
        } else {
            next_gn = gn;
        }

        let ret = self.sppf.find(next_gn, left, right);
        // self.sppf.get_mut(ret).pack_ns.insert(rn);

        let pivot = match ln {
            Some(id) => self.sppf.get(id).ri,
            None => self.sppf.get(rn).li,
        };

        let pack_id = self.sppf.find_pack(ret, gn, pivot, ln, rn);
        self.sppf.get_mut(ret).pack_ns.insert(pack_id);

        ret
    }

    fn du(&mut self, width: u32) {
        let next_gn = self.g_grammar.get(self.gn).seq;
        let leaf_node = self.sppf.find(self.gn, self.i, self.i + width);

        self.dn = Some(self.sppf_update(next_gn, self.dn, leaf_node));
    }

    fn ret(&mut self) {
        let result_node = self.dn.expect("Cannot return with null derivation");
        // Check acceptance
        if self.sn == self.gss_root {
            let lhs_idx = self
                .g_grammar
                .get(self.gn)
                .alt
                .expect("END node must have LHS");
            let start_symbol_idx = self.g_grammar.headers[&self.grammar.start];

            // Check A: Is this the Start Symbol?
            if lhs_idx == start_symbol_idx {
                // Check B: Did we consume all input?
                if self.i as usize == self.input.len() {
                    self.accepting = true;
                }
            }
        }

        {
            let gss_node = self.gss.get_mut(self.sn);
            gss_node.pops.insert(result_node);
        }

        // Get return address and edges to avoid borrow conflict
        let (gn, edges) = {
            let node = self.gss.get(self.sn);
            (node.gn, node.edges.iter().cloned().collect::<Vec<_>>())
        };

        for edge in edges {
            let new_dn = self.sppf_update(gn, edge.sppf_node, result_node);
            self.queue_descriptor(gn, self.i, edge.dst, Some(new_dn));
        }
    }

    pub fn parse_on(&mut self, input: Vec<u32>) -> bool {
        self.initialisation();
        self.input = input;
        self.i = 0;
        self.accepting = false;

        while self.dequeue_descriptor() {
            let current_gnode = self.g_grammar.get(self.gn);

            match current_gnode.kind {
                GKind::Terminal(t_id) => {
                    if (self.i as usize) < self.input.len() && self.input[self.i as usize] == t_id {
                        self.du(1);
                        let next_i = self.i + 1;
                        let next_gn = self.g_grammar.get(self.gn).seq;
                        // Queue descriptor for next position
                        self.queue_descriptor(next_gn, next_i, self.sn, self.dn);
                    }
                }
                GKind::NonTerminal(_nt_id) => {
                    self.call(self.gn);
                }
                GKind::Epsilon => {
                    self.du(0);
                    let next_gn = self.g_grammar.get(self.gn).seq;
                    // Queue descriptor for next position (epsilon doesn't consume input)
                    self.queue_descriptor(next_gn, self.i, self.sn, self.dn);
                }
                GKind::End => {
                    self.ret();
                }
                GKind::LHS(_) | GKind::Alt | GKind::EOS => {
                    // Should not happen during parsing
                    unreachable!();
                }
            }
        }

        self.accepting
    }

    fn sppf_to_tree(&self, root_id: SPPFNodeId) -> Option<ParseTree> {
        let mut visited = HashSet::new();
        self.flatten_tree(root_id, &mut visited).into_iter().next()
    }

    /// Recursive helper to flatten binary SPPF nodes into N-ary lists
    fn flatten_tree(&self, node_id: SPPFNodeId, visited: &mut HashSet<SPPFNodeId>) -> Vec<ParseTree> {
        // Check for cycles to prevent infinite recursion
        if visited.contains(&node_id) {
            return vec![];
        }
        visited.insert(node_id);
        
        let node = self.sppf.get(node_id);
        let g_node = self.g_grammar.get(node.gn);

        // --- Base Case: Leaf Node (Terminal or Epsilon) ---
        if node.pack_ns.is_empty() {
            return match g_node.kind {
                GKind::Terminal(t_id) => {
                    let name = self.grammar.terminal_str(t_id).unwrap_or("?").to_string();
                    vec![ParseTree::leaf(&name)]
                }
                GKind::Epsilon => {
                    vec![ParseTree::leaf("ε")]
                }
                _ => vec![],
            };
        }

        if let Some(pack_id) = node.pack_ns.iter().next() {
            let pack = &self.sppf.pack_nodes[pack_id.0];

            // Collect children from Left and Right subtrees
            let mut children = Vec::new();

            // The Left child (if it exists) represents the prefix of the rule
            if let Some(left_id) = pack.left_c {
                children.extend(self.flatten_tree(left_id, visited));
            }

            // The Right child is the symbol just parsed
            children.extend(self.flatten_tree(pack.right_c, visited));

            // Determine if we Wrap or Flatten
            // ONLY wrap for LHS nodes, NOT for NonTerminal nodes (which are RHS references)
            match g_node.kind {
                GKind::LHS(nt_id) => {
                    let name = self
                        .grammar
                        .non_terminal_str(nt_id)
                        .unwrap_or("?")
                        .to_string();
                    vec![ParseTree::new(ParseSymbol::NonTerminal(name), children)]
                }

                _ => children,
            }
        } else {
            vec![]
        }
    }

    pub fn parse(&mut self, input: &Vec<u32>) -> Option<ParseTree> {
        let accepted = self.parse_on(input.to_vec());
        if accepted {
            let start_nt_id = self.grammar.start;
            let start_gn = *self.g_grammar.headers.get(&start_nt_id).unwrap();

            // Find SPPF node for the start symbol spanning the entire input
            let root_sppf_id = self.sppf.find(start_gn, 0, self.input.len() as u32);

            self.sppf_to_tree(root_sppf_id)
        } else {
            None
        }
    }
}

mod tests {
    use crate::grammars;

    use super::*;

    #[test]
    fn test_grammar_graph_construction() {
        let json = r#"{
            "name": "nested",
            "start": "<S>",
            "rules": {
                "<S>": [["b"], ["a", "<X>", "z"]],
                "<X>": [["x", "<X>"], ["y", "<X>"], []]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let graph_grammar = GrammarGraph::from_numeric(&grammar);

        println!("Grammar Graph Nodes:");
        for (i, node) in graph_grammar.nodes.iter().enumerate() {
            // Print node index, kind and seq/alt indices
            // Also print non-terminals/terminals in string form (not number)
            println!("Node {}: {:?}", i, node);
            let label = match node.kind {
                GKind::LHS(nt_id) => format!(
                    "LHS({:?})",
                    grammar.symbol_to_str(&NumSymbol::NonTerminal(nt_id))
                ),
                GKind::Alt => "ALT".to_string(),
                GKind::Terminal(t_id) => {
                    format!("T({:?})", grammar.symbol_to_str(&NumSymbol::Terminal(t_id)))
                }
                GKind::NonTerminal(nt_id) => format!(
                    "N({:?})",
                    grammar.symbol_to_str(&NumSymbol::NonTerminal(nt_id))
                ),
                GKind::Epsilon => "EPS".to_string(),
                GKind::End => "END".to_string(),
                GKind::EOS => "EOS".to_string(),
            };
            println!(
                "    Kind: {:?}, Seq: {}, Alt: {:?}",
                label, node.seq, node.alt
            );
        }
    }

    #[test]
    fn test_gll_simple() {
        let json = r#"{
            "name": "simple",
            "start": "<S>",
            "rules": {
                "<S>": [["a", "<S>", "b"], []]
            }
        }"#;
        let grammar = grammars::load_grammar_from_str(json).unwrap();
        let mut parser = GLLParser::new(&grammar);

        let token_a = grammar.terminals.get_id("a").expect("Token 'a' not found");
        let token_b = grammar.terminals.get_id("b").expect("Token 'b' not found");
        let input = vec![token_a, token_a, token_b, token_b];

        let parse_tree = parser.parse(&input);
        assert!(parse_tree.is_some(), "Parse tree should be generated");
        println!("{:}", parse_tree.unwrap().display());
    }

    #[test]
    fn test_gll_sexp() {
        // Load grammar from json file in grammars/sexp.json
        let path = "grammars/sexp.json";
        let grammar =
            grammars::load_grammar_from_file(path).expect("Failed to load S-expression grammar");

        // let input = r#"((()..) () () (-4.0.((+8.93 (((((+."wGH").())."`").((.."U"))).((().((-33) (((y/ (() () (((/. ())).())) . ()).(((().(((+1.5.-0).(().+116.36)).(("hDk1".(("b").((..("yk{".()))..))) ((() ("^".()) "Kj") 4.5329 ((((..(/ -55)).((().(())).(((-00."vl") +6.08))))).(-70050.2.(((/.((() (/8)) ((...).((().(()."9Y")).())))).()).(85797.03.(((p.(((().("b".())).(().(((-17.0.("|")).*j).())))."HMIY")).(-9.40..)).()))))))))).((().00).())).(/.(().((().(9.50.("Tq)".(((..((+2..).((()."K")))).920) (-2))))).((((*.)..)).(((((..++)).((. ("dIF") "[" -98 (())).((().((((().("J"..)).-1523)).(((((-.(+7.4.((())))) 1).()))..)))..-)))."y)D")..))))))).()))).060.68)) ()).e2)) . r* .)"#;
        let input = r#"(.)"#;
        let tokens: Vec<u32> = grammar.tokenize(input).expect("Failed to tokenize input");

        println!("\n=== Testing GLL Parser (S-expression) ===");
        let mut parser = GLLParser::new(&grammar);
        let result = parser.parse(&tokens);
        match &result {
            Some(tree) => {
                println!("\n✓ Parse successful!");
                println!("{}", tree.display());
            }
            None => println!("\n✗ Parse failed!"),
        }
    }
}
