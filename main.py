import numpy as np
import random
import math
import time
import copy

def distance(x1, x2, y1, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


def Setting(FILENAME):
    mat = []
    with open('/home/kurozumi/デスクトップ/benchmark/'  + FILENAME, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            row = []
            toks = line.split(' ')
            for tok in toks:
                try:
                    num = float(tok)
                except ValueError:
                    continue
                row.append(num)
            mat.append(row)
    # インスタンスの複数の行（問題設定）を取り出す
    Setting_Info = mat.pop(0)  # 0:車両数、4:キャパシティ、8:一台あたりの最大移動時間(min)、9:一人あたりの最大移動時間(min)

    # デポの座標を取り出す
    depo_zahyo = np.zeros(2)  # デポ座標配列
    x = mat.pop(-1)
    depo_zahyo[0] = x[1]
    depo_zahyo[1] = x[2]

    request_number = len(mat) - 1

    # 各距離の計算
    c = np.zeros((len(mat), len(mat)), dtype=float, order='C')

    # eがtime_windowの始、lが終
    e = np.zeros(len(mat), dtype=float, order='C')
    l = np.zeros(len(mat), dtype=float, order='C')

    # テキストファイルからtime_windowを格納 & 各ノードの距離を計算し格納
    for i in range(len(mat)):
        e[i] = mat[i][5]
        l[i] = mat[i][6]
        for j in range(len(mat)):
            c[i][j] = distance(mat[i][1], mat[j][1], mat[i][2], mat[j][2])

    # 乗り降りの0-1情報を格納
    noriori = np.zeros(len(mat), dtype=int, order='C')
    for i in range(len(mat)):
        noriori[i] = mat[i][4]

    return Setting_Info, request_number, depo_zahyo, c, e, l, noriori

def Route_cost(route):
    Route_sum = 0
    Route_sum_k = np.zeros(len(route), dtype=float, order='C')
    for i in range(len(route)):
        if len(route[i]) == 0:
            Route_sum_k[i] = 0
        else:
            for j in range(len(route[i]) - 1):
                Route_sum_k[i] = Route_sum_k[i] + c[route[i][j]][route[i][j + 1]]
            Route_sum_k[i] = Route_sum_k[i] + c[0][route[i][0]]
            Route_sum_k[i] = Route_sum_k[i] + c[0][route[i][j + 1]]
        Route_sum = Route_sum + Route_sum_k[i]

    return Route_sum


def route_k_cost_sum(route_k):
    route_k_sum = 0
    for i in range(len(route_k) - 1):
        route_k_sum = route_k_sum + c[route_k[i]][route_k[i + 1]]
    route_k_sum = route_k_sum + c[0][route_k[0]]
    route_k_sum = route_k_sum + c[0][route_k[i + 1]]

    return route_k_sum


def capacity(route):
    q = np.zeros(int(m), dtype=int, order='C')
    capacity_over = 0
    for i in range(len(route)):
        for j in range(len(route[i])):
            q[i] = q[i] + noriori[route[i][j]]
            if q[i] > Q_max:
                capacity_over += 1
    return capacity_over


def capacity_route_k(route_k):
    capacity_over = 0
    q = 0
    for i in range(len(route_k)):
        q = q + noriori[route_k[i]]
        if q > Q_max:
            capacity_over += 1
    return capacity_over


def time_caluculation(Route_k):
    B = np.zeros(n + 2, dtype=float, order='C')  # サービス開始時間（e.g., 乗せる時間、降ろす時間)
    A = np.zeros(n + 2, dtype=float, order='C')  # ノード到着時間
    D = np.zeros(n + 2, dtype=float, order='C')  # ノード出発時間
    W = np.zeros(n + 2, dtype=float, order='C')  # 車両待ち時間
    L = np.zeros(int(n / 2), dtype=float, order='C')  # リクエストiの乗車時間
    if not len(Route_k) == 0:
        for i in range(len(Route_k)):
            if i == 0:
                A[Route_k[i]] = D[i] + c[i][Route_k[i]]
                B[Route_k[i]] = max(e[Route_k[i]], A[Route_k[i]])
                D[Route_k[i]] = B[Route_k[i]] + d
                W[Route_k[i]] = B[Route_k[i]] - A[Route_k[i]]
            else:
                A[Route_k[i]] = D[Route_k[i - 1]] + c[Route_k[i - 1]][Route_k[i]]
                B[Route_k[i]] = max(e[Route_k[i]], A[Route_k[i]])
                D[Route_k[i]] = B[Route_k[i]] + d
                W[Route_k[i]] = B[Route_k[i]] - A[Route_k[i]]
        A[-1] = D[Route_k[-1]] + c[0][Route_k[-1]]
        B[-1] = A[-1]
        for i in range(len(Route_k)):
            if Route_k[i] <=  n/ 2:
                L[Route_k[i] - 1] = B[Route_k[i] + int(n/ 2)] - D[Route_k[i]]
    return A, B, D, W, L


def time_window_penalty(route_k, B):  # 論文でのw(s)
    sum = 0
    for i in range(len(route_k)):
        a = B[route_k[i]] - l[route_k[i]]
        if a > 0:
            sum = sum + a
    a = B[-1] - l[0]
    if a > 0:
        sum = sum + a
    return sum


def ride_time_penalty(L):  # 論文でのt_s
    sum = 0
    for i in range(len(L)):
        a = L[i] - L_max
        if a > 0:
            sum = sum + a
    return sum

def penalty_sum(route):
    parameta = np.zeros(4)
    c_s = Route_cost(route)
    q_s = capacity(route)
    d_s = 0
    w_s = 0
    t_s = 0
    for i in range(len(route)):
        if not len(route[i]) == 0:
            ROUTE_TIME_info = time_caluculation(route[i], n)
            d_s_s = (ROUTE_TIME_info[1][-1]-ROUTE_TIME_info[1][route[i][1]] +c[0][route[i][1]])- T_max
            if d_s_s < 0:
                d_s_s = 0
            d_s = d_s + d_s_s
            w_s = w_s + time_window_penalty(route[i], ROUTE_TIME_info[1])
            t_s = t_s + ride_time_penalty(ROUTE_TIME_info[4])
    no_penalty = c_s + q_s + d_s + w_s + t_s

    return no_penalty

def Set_Weight(alpha,beta,gamma,delta,step):
    weight = np.array([alpha,beta**(1/step),gamma,delta])
    return weight


def Greedy_aco(step,nants,p):
    x_best = [[] * 1 for i in range(m)]
    Tau = np.zeros((step,n+1,n+1))
    Eta = np.zeros((n+1,n+1))
    t = 0
    while True:
        if t == step:
            break
        weight = Set_Weight(1,0.02,100,1,t)
        # Tau(t+1) = ここでフェロモン更新
        Repeat = np.zeros((n+1,n+1)) # 初期化
        quantity = np.ones(n+1) # 初期化
        ant =1
        while True:
            if ant > nants:
                break
            P = quantity/ant

            x = Construction_solution(weight,Tau,Eta,Repeat,P)

            """
            Construction ant solution x
            decision 
                weightage Weight
                current pheromone Tau
                heuristic infomation Eta
                repeat counter R
                quantity counter Q
            """
            if penalty_sum(x) == 0 and x < x_best:
                x_best = x
            """
            Update
                best known solution x_best
                next pheromone Tau
                repeat counter R
                quantity counter Q
            """
            ant +=1

        t +=1
def Construction_solution(weight,Tau,Eta,R,P):
    x = [[] * 1 for i in range(m)]
    """
    移動セットAをつくる(どう作ればいいのかわからないので保留)
    """
    while True:
        #以下初期化
        #ここでx_best = []
        S_highest = -1.
        highest_set = np.zeros(2)

        #車両_k_の現在地を取得
        for k in range(3):
            if len(x[k]) == 0:
                i= 0
            else:
                i = x[k][-1]
            for j in node_set:
                pass

    pass

if __name__ == '__main__':
    FILENAME = 'darp01.txt'
    Setting_Info = Setting(FILENAME)[0]

    n = int(Setting(FILENAME)[1])  # depoを除いたノード数
    m = int(Setting_Info[0])  # 車両数
    d = 5  # 乗り降りの時間
    Q_max = Setting_Info[4]  # 車両の最大容量 global変数 capacity関数で使用
    T_max = Setting_Info[8]  # 一台当たりの最大移動時間
    L_max = Setting_Info[9]  # 一人あたりの最大移動時間

    noriori = np.zeros(n + 1, dtype=int, order='C')
    noriori = Setting(FILENAME)[6]  # global変数  capacity関数で使用

    depo_zahyo = Setting(FILENAME)[2]  # デポの座標

    c = np.zeros((n + 1, n + 1), dtype=float, order='C')
    c = Setting(FILENAME)[3]  # 各ノード間のコスト

    e = np.zeros(n + 1, dtype=float, order='C')
    l = np.zeros(n + 1, dtype=float, order='C')
    e = Setting(FILENAME)[4]
    l = Setting(FILENAME)[5]

    pick_up = np.arange(1,int(n/2)+1)
    drop_off = np.arange(int(n/2)+1,n+1)
    node_set = np.arange(1,n+1)
    #Greedy_aco(5,4,3)
    print(node_set)