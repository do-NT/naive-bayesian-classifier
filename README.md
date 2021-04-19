---


---

<p><strong>1. Bài toán</strong><br>
Khi nhận được hình ảnh của một bông hoa lan iris, từ việc đo đạc các kích thước của nó chúng ta cần đoán xem nó thuộc loài hoa nào.<br>
<img src="https://machinelearningcoban.com/assets/knn/iris.png" alt=""><br>
Đây là bài toán thuộc dạng phân loại nhiều lớp (multiclass classification). Số lượng record cho từng class trong dataset là cân bằng. Có tất cả 150 records với mỗi record là 4 biến input (coi như một vector 4 chiều) và 1 biến output. Tên các biến được liệt kê bên dưới:</p>
<ul>
<li>Chiều dài đài hoa (cm)</li>
<li>Chiều rộng đài hoa (cm)</li>
<li>Chiều dài cánh hoa (cm)</li>
<li>Chiều rộng cánh hoa (cm)</li>
</ul>
<p>Ví dụ một mẫu gồm 5 record:</p>
<pre><code>5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
</code></pre>
<p><strong>2. Hướng giải quyết</strong><br>
Vì các đại lượng input đều là các đại lượng liên tục, cho nên nếu phải giải quyết bài toán bằng Multinomial naive Bayes thì cần phải xử lý data bằng cách phân loại các input vào các khoảng do mình định đoạt. Việc định đoạt các khoảng như vậy có thể ảnh hưởng tới kết quả cuối cùng của bài toán, do đó việc xác định khoảng sao cho hiệu quả là việc tốn thời gian. Do đó với bài toán này, mình muốn sử dụng Gaussian naive Bayes để xử lý, mình sẽ giả sử rằng các đại lượng input được phân phối theo phân phối chuẩn.<br>
Công thức cốt lõi:</p>
<blockquote>
<p>P(class|data) = (P(data|class) * P(class)) / P(data)<br>
=&gt;    P(class|data) ~ (P(data|class) * P(class))</p>
</blockquote>
<p><strong>3. Các bước giải quyết bài toán</strong><br>
<em><strong>a) Phân nhóm data</strong></em></p>
<pre><code>def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
</code></pre>
<p><em><strong>b) Tóm lược dataset: Lấy các tham số của từng phân phối</strong></em><br>
Tính toán kì vọng và độ lệch chuẩn của từng cột trong dataset<br>
<em>Kì vọng:</em></p>
<pre><code>def mean(numbers):
	return sum(numbers)/float(len(numbers))
</code></pre>
<p><em>Độ lệch chuẩn:</em></p>
<pre><code>from math import sqrt

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
</code></pre>
<p><em>Tổng hợp:</em></p>
<pre><code>def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
</code></pre>
<p><em><strong>c) Tóm lược data theo từng class</strong></em><br>
Sử dụng hàm <code>separate_by_class(dataset)</code>để chia dataset ban đầu thành các datasets nhỏ hơn được phân loại theo class (loài hoa). Sau đó sử dụng hàm <code>summarize_dataset(dataset)</code> để tóm lược data theo từng class.<br>
<em><strong>d) Xây dựng hàm mật độ xác suất Gaussian</strong></em><br>
<em>Hàm mật độ xác suất:</em><br>
<img src="https://scontent-sin6-1.xx.fbcdn.net/v/t1.15752-9/175892299_731468304190132_2902233191752787924_n.png?_nc_cat=109&amp;ccb=1-3&amp;_nc_sid=ae9488&amp;_nc_ohc=rJamFW78QB8AX8LJm3W&amp;_nc_ht=scontent-sin6-1.xx&amp;oh=f86c1eaab58f044938b70ffb2b4bce0f&amp;oe=60A1C801" alt="Không có mô tả."></p>
<pre><code># Tính giá trị của hàm mật độ xác suất với input là x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
</code></pre>
<p><em><strong>e) Tính toán các xác suất</strong></em><br>
Áp dụng tư tưởng “naive”:</p>
<blockquote>
<p>Thay vì tính<br>
P(class|data) = (P(data|class) * P(class)) / P(data)<br>
Ta tính<br>
P(class=0|X1,X2) = P(X1|class=0) <em>…</em> P(Xn|class=0) * P(class=0)</p>
</blockquote>
<pre><code># calculate the probabilities of predicting each class for a given row  
# P(class|data)  
def calculate_class_probabilities(summaries, row):  
    total_rows = sum([summaries[label][0][2] for label in summaries])  
    probabilities = dict()  
    for class_value, class_summaries in summaries.items():  
        # P(class)  
  probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)  
        for i in range(len(class_summaries)):  
            mean, stdev, _ = class_summaries[i]  
            # multiply by P(data|class) = P(x1|class). ... .P(xn|class)  
 # row[i] ~ xi  probabilities[class_value] *= calculate_probability(row[i], mean, stdev)  
    return probabilities
</code></pre>
<p>Sử dụng k-folds cross validation với k = 5 để kiểm tra độ chính xác của model, cho thấy hiệu quả tốt:</p>
<pre><code>Scores: [93.33333333333333, 96.66666666666667, 100.0, 93.33333333333333, 93.33333333333333]
Mean Accuracy: 95.333%
</code></pre>

