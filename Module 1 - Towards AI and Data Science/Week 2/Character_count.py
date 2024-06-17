'''
2. Thực hiện theo các yêu cầu sau.
Viết function trả về một dictionary đếm số lượng chữ xuất hiện trong một từ, với key là chữ cái
và value là số lần xuất hiện
• Input: một từ
• Output: dictionary đếm số lần các chữ xuất hiện
• Note: Giả sử các từ nhập vào đều có các chữ cái thuộc [a-z] hoặc [A-Z]
'''

def count_chars(word):
    result = {}
    for char in word:
        result[char] = result.get(char, 0) + 1
    return result

# Example:
print(count_chars('smiles')) # Output: {'s': 2, 'm': 1, 'i': 1, 'l': 1, 'e': 1}