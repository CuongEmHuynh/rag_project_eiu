# Contribution RAG Pipeline
1. Từ nguồn dữ liệu hiện tại thực nghiệm các phương pháp chunking, thiết kế có thể switch qua lại các phương pháp tối đa 4. 
2. Sau khi đã chunking xong, lựa chọn tìm hiểu các model phục vụ cho việc embedding có thể switch qua lại các model để tìm ra phương pháp chunking nào tốt nhất đối với model embedding nào. Ưu tiên chọn chunking với length dài nhất để đưa vào CSDL Qdrant Vector database.
3. Tách knowledge base ra làm 2 phần
    - Chỉ sử dụng CSDL Qdrant
    - Kết hợp thêm CSDL Graph Database
    => Có thể tuỳ biết tắt bật khi thực hiện query 
4. Tìm kiếm 4,5 Model LLM sắp xếp theo thứ tự cao đến thấp. Tham số, thông số nằm trong khoản 3B, 6B, 8B. Bao gồm các thông số, các model retrain, có hỗ trợ đa ngôn ngữ trong đó có Tiếng Việt
    - Các model top như: LLma, Deepseek, Qwen, Mistral...
    - Có thể tuỳ biến chọn các Model 
5. Thực nghiệm tìm ra 1 bộ có kết quả tốt nhất để demo báo cáo Thầy.
# Contribution Matrix Evulation Pipeline
1. Tìm các bài báo chứa bộ đánh giá
2. Lọc và chọn ra các phương pháp tốt nhất để đánh gía.
