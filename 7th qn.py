import numpy as np
from itertools import combinations

class MarketBasketAnalysis:
    def __init__(self, min_supp=0.5, min_conf=0.7):
        """
        Initialize parameters for the Apriori-based market basket analysis.
        
        Parameters:
        - min_supp: Minimum support threshold (float between 0 and 1)
        - min_conf: Minimum confidence threshold (float between 0 and 1)
        """
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.freq_patterns = None
        self.rules = None
    
    def _compute_support(self, item_grp, dataset):
        """Calculate support for a given item group."""
        count = 0
        for record in dataset:
            if all(element in record for element in item_grp):
                count += 1
        return count / len(dataset)
    
    def _generate_combinations(self, itemsets, size):
        """Generate potential item combinations of given size."""
        items = sorted(set(element for subset in itemsets for element in subset))
        return list(combinations(items, size))
    
    def _filter_combinations(self, candidate_sets, prev_patterns, size):
        """Filter out invalid candidates based on Apriori principle."""
        valid_sets = []
        for candidate in candidate_sets:
            subsets = combinations(candidate, size-1)
            if all(subset in prev_patterns for subset in subsets):
                valid_sets.append(candidate)
        return valid_sets
    
    def analyze(self, dataset):
        """
        Discover frequent itemsets and generate association rules.
        
        Parameters:
        - dataset: List of transactions (each transaction is a list of items)
        """
        # Convert dataset transactions to immutable sets
        dataset = [frozenset(transaction) for transaction in dataset]
        
        # Identify frequent itemsets
        self.freq_patterns = {}
        size = 1
        
        # Find frequent single-item sets
        elements = set(item for record in dataset for item in record)
        candidates = [frozenset([item]) for item in elements]
        
        while candidates:
            # Compute support for each candidate
            valid_candidates = []
            for candidate in candidates:
                support = self._compute_support(candidate, dataset)
                if support >= self.min_supp:
                    valid_candidates.append((candidate, support))
            
            # Store frequent itemsets of the current size
            if valid_candidates:
                self.freq_patterns[size] = valid_candidates
                size += 1
                
                # Generate next level candidates
                candidates = self._generate_combinations(
                    [itemset for itemset, support in valid_candidates], 
                    size
                )
                
                # Apply pruning step
                previous_sets = set(itemset for itemset, support in valid_candidates)
                candidates = self._filter_combinations(candidates, previous_sets, size-1)
            else:
                candidates = None
        
        # Construct association rules
        self.rules = []
        for size, patterns in self.freq_patterns.items():
            if size == 1:
                continue
                
            for itemset, support in patterns:
                # Generate non-empty subsets for rule formation
                for i in range(1, size):
                    for left_side in combinations(itemset, i):
                        left_side = frozenset(left_side)
                        right_side = itemset - left_side
                        
                        # Compute confidence metric
                        left_support = self._compute_support(left_side, dataset)
                        confidence = support / left_support
                        
                        if confidence >= self.min_conf:
                            self.rules.append({
                                'lhs': left_side,
                                'rhs': right_side,
                                'support': support,
                                'confidence': confidence
                            })
    
    def get_frequent_patterns(self, size=None):
        """
        Retrieve frequent itemsets.
        
        Parameters:
        - size: Specific size of itemsets to return (None for all sizes)
        
        Returns:
        - List of frequent itemsets along with their support values
        """
        if size is None:
            return [pattern for level in self.freq_patterns.values() for pattern in level]
        return self.freq_patterns.get(size, [])
    
    def get_association_rules(self):
        """Retrieve association rules that satisfy the minimum confidence constraint."""
        return self.rules

# Example execution
if __name__ == "__main__":
    # Sample market basket transactions
    dataset = [
        ['bread', 'milk'],
        ['bread', 'diapers', 'beer', 'eggs'],
        ['milk', 'diapers', 'beer', 'cola'],
        ['bread', 'milk', 'diapers', 'beer'],
        ['bread', 'milk', 'diapers', 'cola']
    ]
    
    # Instantiate and execute analysis
    mba = MarketBasketAnalysis(min_supp=0.4, min_conf=0.6)
    mba.analyze(dataset)
    
    # Display results
    print("Frequent Itemsets:")
    for size, patterns in mba.freq_patterns.items():
        print(f"\nSize {size}:")
        for pattern, support in patterns:
            print(f"{set(pattern)}: support = {support:.2f}")
    
    print("\nAssociation Rules:")
    for rule in mba.get_association_rules():
        print(f"{set(rule['lhs'])} => {set(rule['rhs'])} "
              f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f})")
