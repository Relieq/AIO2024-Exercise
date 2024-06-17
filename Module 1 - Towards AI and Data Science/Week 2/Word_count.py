'''
3. Thực hiện theo các yêu cầu sau.
Viết function đọc các câu trong một file txt, đếm số lượng các từ xuất hiện và trả về một dictionary
với key là từ và value là số lần từ đó xuất hiện.
• Input: Đường dẫn đến file txt
• Output: dictionary đếm số lần các từ xuất hiện
• Note:
    - Giả sử các từ trong file txt đều có các chữ cái thuộc [a-z] hoặc [A-Z]
    - Không cần các thao tác xử lý string phức tạp nhưng cần xử lý các từ đều là viết
thường
'''
def count_word(file_path):
    counter = {}
    with open(file_path, 'r') as file:
        for line in file:
            words = line.strip().split()
            for word in words:
                word = word.lower()
                counter[word] = counter.get(word, 0) + 1
    # Sắp xếp counter theo thứ tự từ điển
    counter = dict(sorted(counter.items()))
    return counter

# Example:
file_path = 'Module 1 - Towards AI and Data Science\Week 2\P1_data.txt'
print(count_word(file_path))