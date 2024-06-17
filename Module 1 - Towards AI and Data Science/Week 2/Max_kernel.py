'''
1. Cho một list các số nguyên num_list và một sliding window có kích thước size k di
chuyển từ trái sang phải. Mỗi lần dịch chuyển 1 vị trí sang phải có thể nhìn thấy
đươc k số trong num_list và tìm số lớn nhất trong k số này sau mỗi lần trượt k phải
lớn hơn hoặc bằng 1.

Input: num_list = [3, 4, 5, 1, -44 , 5 ,10, 12 ,33, 1] với k=3
Output: [5, 5, 5, 5, 10, 12, 33, 33]
Ví dụ:
[3, 4, 5], 1, -44 , 5 ,10, 12 ,33, 1 => max 5
3, [4, 5, 1], -44 , 5 ,10, 12 ,33, 1 => max 5
3, 4, [5, 1, -44] , 5 ,10, 12 ,33, 1 => max 5
3, 4, 5, [1, -44 , 5] ,10, 12 ,33, 1 => max 5
3, 4, 5, 1, [-44 , 5 ,10], 12 ,33, 1 => max 10
3, 4, 5, 1, -44 , [5 ,10, 12] ,33, 1 => max 12
3, 4, 5, 1, -44 , 5 ,[10, 12 ,33], 1 => max 33
3, 4, 5, 1, -44 , 5 ,10, [12 ,33, 1] => max 33
'''

def max_kernel(num_list, k):
    result = []
    for i in range(len(num_list) - k + 1):
        result.append(max(num_list[i:i+k]))
    return result

# Example:
num_list = [3, 4, 5, 1, -44 , 5 ,10, 12 ,33, 1]
k = 3
print(max_kernel(num_list, k)) # Output: [5, 5, 5, 5, 10, 12, 33, 33]