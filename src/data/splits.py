"""Default roles for new experiments; historical manifests remain immutable."""


def train_test_role(role):
    """Preserve declared training membership; pool all other roles into test."""
    return 'train' if role == 'train' else 'test'
