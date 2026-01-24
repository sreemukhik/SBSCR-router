"""
Train a Decision Tree Classifier to find optimal routing thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sbscr.evaluation.datasets import load_all_datasets
from sbscr.core.metadata import ComplexityExtractor

def train():
    print("🚀 Starting Data-Driven Calibration...")
    
    # 1. Load Data
    datasets = load_all_datasets()
    
    # Load Synthetic Data
    try:
        with open("data/synthetic_dataset.json", "r") as f:
            synthetic = json.load(f)
            # Add to proper dataset categories
            datasets['synthetic'] = synthetic
            print(f"✅ Loaded {len(synthetic)} synthetic examples")
    except FileNotFoundError:
        print("⚠️  Warning: Synthetic dataset not found. Using default only.")
    extractor = ComplexityExtractor()
    
    data = []
    
    # Flatten datasets into a training list
    for dataset_name, queries in datasets.items():
        for item in queries:
            query = item['query']
            expected_model = item['expected_model']
            
            # Extract features (Simulating what router sees at runtime)
            features = extractor.extract_features(query)
            complexity = extractor.estimate_complexity(query)
            
            # Simplified feature set for the tree
            # We want rules like: "if complexity > X and domain is Y..."
            row = {
                'complexity': complexity,
                'is_code_domain': 1 if features['domain'] == 'code' else 0,
                'is_math_domain': 1 if features['domain'] == 'math' else 0,
                'is_creative_domain': 1 if features['domain'] == 'creative' else 0,
                'word_count': features['word_count'],
                'target': expected_model
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    print(f"📊 Loaded {len(df)} training examples")
    
    # 2. Prepare X and y
    feature_cols = ['complexity', 'is_code_domain', 'is_math_domain', 'is_creative_domain', 'word_count']
    X = df[feature_cols]
    y = df['target']
    
    # 3. Train Decision Tree
    # max_depth=4 to keep rules simple & readable
    clf = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    
    # 4. Evaluate (On Training Data - Overfitting is fine, we want to memorize rules for now)
    score = clf.score(X, y)
    print(f"🎯 Training Accuracy (Model Consistency): {score:.1%}")
    
    # 5. Export Rules
    print("\n📜 EXTRACTED RULES (Copy these logic into sbscr.py):\n")
    tree_rules = export_text(clf, feature_names=feature_cols)
    print(tree_rules)
    
    # 6. Structured Rule Export (Helper to read)
    print("\n💡 INTERPRETATION:\n")
    
    # Walk the tree to show leaf paths (Simple visualization)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    classes = clf.classes_

    def recurse(node, depth):
        indent = "  " * depth
        if children_left[node] != children_right[node]:
            fname = feature_cols[feature[node]]
            th = threshold[node]
            print(f"{indent}If {fname} <= {th:.2f}:")
            recurse(children_left[node], depth + 1)
            print(f"{indent}Else ({fname} > {th:.2f}):")
            recurse(children_right[node], depth + 1)
        else:
            # Leaf
            class_idx = np.argmax(value[node])
            class_name = classes[class_idx]
            print(f"{indent}👉 RETURN '{class_name}'")

    recurse(0, 0)

if __name__ == "__main__":
    train()
