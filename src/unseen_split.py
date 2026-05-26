def split_seen_unseen(train_patterns, test_patterns):
    train_vocabulary = set(train_patterns)

    seen = []
    unseen = []

    for pattern in test_patterns:
        if pattern in train_vocabulary:
            seen.append(pattern)
        else:
            unseen.append(pattern)

    return seen, unseen
