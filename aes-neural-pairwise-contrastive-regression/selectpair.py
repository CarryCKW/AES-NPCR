def score_groups_divide(X_train, Y_train):
    essay_nums = X_train.shape[0]
    scores = set()
    for i in range(Y_train.shape[0]):
        scores.add(str(Y_train[i][0]))
    score_groups_num = len(scores)

    score_ids = {}
    scores = list(scores)
    for l in range(len(scores)):
        score_ids[scores[l]] = []

    for j in range(X_train.shape[0]):
        score_ids[str(Y_train[j][0])].append(j)

    # tips:
    # 1. score_ids: {'8.0': [2, 7, 8, 12], '10.0': [...], '7.0': [...], ...}
    # -> {'score':[essay_id_list]}
    # 2. scores: ['8.0', '10.0', '7.0', ...], a list
    # 3. score_groups_num equals to len(scores)
    return essay_nums, score_groups_num, scores, score_ids
