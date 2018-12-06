import numpy as np


# 輸入Lab,e_range: delta e的容忍範圍, number:想要取的點位數量
def find_delta_e_range(L, a, b, e_range, number):
    init = np.arange(number * 3).reshape(number, 3)
    init = init.astype(np.float32)
    count = 0

    while count < number:
        # 在+-e_range內隨機產生一組1*3array [1, 1.3, 2]
        e_range_array = np.random.randint(low=(-1) * e_range * 100, high=e_range * 100, size=3)
        e_range_array = e_range_array / 100

        # 若該array平方相加開更號<e_range, 則把Lab加上這組array,放置到init,count++
        if (e_range_array[0] ** 2 + e_range_array[1] ** 2 + e_range_array[1] ** 2) <= e_range ** 2:
            init[count, 0] = L + e_range_array[0]
            init[count, 1] = a + e_range_array[1]
            init[count, 2] = b + e_range_array[2]
            count += 1
    return init
