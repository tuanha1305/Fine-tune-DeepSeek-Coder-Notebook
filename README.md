# DeepSeek Coder Fine-tuning

Project này giúp fine-tune mô hình DeepSeek Coder 6.7B trên Google Colab, được tối ưu hóa cho hiệu suất và sử dụng bộ nhớ.

## Mô tả

DeepSeek Coder là một mô hình ngôn ngữ lớn được huấn luyện đặc biệt cho lập trình. Project này cung cấp một pipeline hoàn chỉnh để fine-tune mô hình với dữ liệu của riêng bạn, được tối ưu hóa cho môi trường Google Colab.

### Các tính năng chính

- Fine-tune mô hình DeepSeek Coder 6.7B
- Tối ưu hóa cho Google Colab (GPU/TPU)
- Hỗ trợ xử lý dữ liệu hiệu quả
- Quản lý bộ nhớ thông minh
- Lưu và tải mô hình fine-tuned

## Yêu cầu hệ thống

- Google Colab (khuyến nghị dùng Colab Pro+ với GPU A100)
- Python 3.8+
- PyTorch 2.0+
- CUDA support

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/tuanha1305/Fine-tune-DeepSeek-Coder-Notebook.git
cd Fine-tune-DeepSeek-Coder-Notebook
```

2. Upload notebook lên Google Colab hoặc chạy locally.

3. Cài đặt các dependencies:
```python
!pip install transformers datasets torch accelerate bitsandbytes
```

## Chuẩn bị dữ liệu

Dữ liệu training cần được chuẩn bị ở định dạng JSON với cấu trúc sau:

```json
[
  {
    "instruction": "Write a Python function to sort a list",
    "output": "def sort_list(lst):\n    return sorted(lst)"
  },
  {
    ...
  }
]
```

## Sử dụng

1. Mở `Fine-tune-DeekSeek-Coder.ipynb` trong Google Colab

2. Chọn Runtime với GPU:
   - Runtime > Change runtime type
   - Hardware accelerator > GPU
   - GPU type > A100 (nếu có)

3. Upload dữ liệu training

4. Điều chỉnh các hyperparameter trong cell cuối:
```python
train(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    data_path="path_to_your_data.json",
    output_dir="./deepseek_coder_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5
)
```

## Các tham số quan trọng

- `model_name`: Tên hoặc đường dẫn đến pretrained model
- `data_path`: Đường dẫn đến file dữ liệu training JSON
- `output_dir`: Thư mục lưu model sau khi fine-tune
- `per_device_train_batch_size`: Kích thước batch cho mỗi GPU
- `gradient_accumulation_steps`: Số bước tích lũy gradient
- `num_train_epochs`: Số epoch training
- `learning_rate`: Tốc độ học

## Quản lý bộ nhớ

Project được tối ưu hóa để chạy trên Google Colab:
- Sử dụng 8-bit quantization
- Gradient checkpointing
- Efficient memory handling
- Batch size tự động điều chỉnh

## Lưu ý

- Nên sử dụng GPU A100 để có hiệu suất tốt nhất
- Monitor VRAM usage để tránh OOM errors
- Lưu model checkpoint thường xuyên
- Kiểm tra dữ liệu training trước khi bắt đầu

## Contributing

Mọi đóng góp đều được hoan nghênh. Vui lòng:
1. Fork project
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License
- MIT

## Tác giả
- [Lucas Ha](https://github.com/tuanha1305)

## Liên hệ

- Email: tuanictu97@gmail.com
- GitHub: tuanha1305