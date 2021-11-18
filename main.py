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

def Greedy_aco(step,nants,p):
    x_best = [[] * 1 for i in range(m)]
    Tau = np.zeros((step,n+1,n+1))
    Eta = np.zeros((n+1,n+1))
    t = 0
    while True:
        if t == step:
            break
        # Weight = ここで重みを更新（初期化）
        # Tau(t+1) = ここでフェロモン更新
        Repeat = np.zeros((n+1,n+1)) # 初期化
        quantity = np.ones(n+1) # 初期化
        ant =1
        while True:
            if ant > nants:
                break
            P = quantity/ant
            """
            Construction ant solution x
            decision 
                weightage Weight
                current pheromone Tau
                heuristic infomation Eta
                repeat counter R
                quantity counter Q
            """

            """
            Update
                best known solution x_best
                next pheromone Tau
                repeat counter R
                quantity counter Q
            """
            ant +=1

        t +=1
#def Construction_solution():

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

    Greedy_aco(5,4,3)
