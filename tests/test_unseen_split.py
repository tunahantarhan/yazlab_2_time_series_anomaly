from src.unseen_split import split_seen_unseen


def test_split_seen_unseen_patterns():
    train_patterns = ["aaa", "aab", "abc"]

    test_patterns = ["aaa", "bbb", "abc", "ccc"]

    seen, unseen = split_seen_unseen(
        train_patterns,
        test_patterns
    )

    assert seen == ["aaa", "abc"]
    assert unseen == ["bbb", "ccc"]
