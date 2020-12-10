'''
a script to calculate the advantage of batch process of Re-identification
'''


def cal_C(all, n):
    result = 1
    for i in range(n):
        result = result * all / (i + 1)
        all = all - 1
    return result


def main():
    m = int(input("Total num of the database: \n"))
    n = int(input("Match threshold num: \n"))
    p = float(input("mAP(0~1.0): \n"))
    wrong_prob = 0
    for i in range(n):
        wrong_prob = wrong_prob + cal_C(m, i) * ((1 - p)**(m - i)) * (p**i)
    print(wrong_prob)


if __name__ == "__main__":
    main()