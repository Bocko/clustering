def adjToCluster(adj):
    n = len(adj)
    mat = [[0 for i in range(n)]]

    for i in range(n):
        if sum(adj[i]) == 0:
            mat[0][i] = 1

    k = 1
    while sum(mat[k-1]) != 0:
        mat.append([0 for i in range(n)])
        a = [i for i, x in enumerate(mat[k-1]) if x == 1]
        for i in range(n):
            for j in range(len(a)):
                if sum(diffValLs(i,a)) == len(a) and adj[i][a[j]] == 1:
                    mat[k][i] = 1

        b = [i for i, x in enumerate(mat[k]) if x == 1]
        for i in range(len(b)):
            for j in range(len(b)):
                if i != j and adj[b[i]][b[j]] == 1:
                    mat[k][b[i]] = 0

        k += 1

    group_nb = k-2
    result = [0 for i in range(n)]
    l = 0
    for i in range(group_nb,-1,-1):
        a = [i for i, x in enumerate(mat[i]) if x != 0]
        for ind in a:
            result[ind] = group_nb - i #+ 1

    return result



def diffValLs(val,ls):
    diff = []
    for elem in ls:
        if elem != val:
            diff.append(1)
        else:
            diff.append(0)
    return diff
