# Hướng dẫn cài đặt các container cơ sở dữ liệu cho RAG

Tài liệu này dùng để khởi tạo các dịch vụ cơ sở dữ liệu trong thư mục `settings` phục vụ hệ thống RAG hiện tại. Theo cấu hình trong [docker-compose.yaml](/e:/Master/Project_master/rag_project_eiu/settings/docker-compose.yaml), dự án đang sử dụng:

- `Qdrant`: vector database để lưu embedding và tìm kiếm ngữ nghĩa.
- `Neo4j`: graph database để lưu tri thức dạng node/relationship.

## 1. Điều kiện cần

Máy cần cài sẵn:

- Docker Desktop hoặc Docker Engine
- Docker Compose

Kiểm tra nhanh:

```powershell
docker --version
docker compose version
```

## 2. Cấu trúc thư mục dữ liệu

Các thư mục volume đã được map sẵn để lưu dữ liệu bền vững:

- `settings/qdrant/storage` -> dữ liệu của Qdrant
- `settings/neo4j/data` -> dữ liệu database của Neo4j
- `settings/neo4j/logs` -> log của Neo4j
- `settings/neo4j/plugins` -> plugin Neo4j, hiện đang có `apoc.jar`

Nếu clone mới dự án mà các thư mục này chưa có, có thể tạo trước:

```powershell
mkdir settings\qdrant\storage
mkdir settings\neo4j\data
mkdir settings\neo4j\logs
mkdir settings\neo4j\plugins
```

## 3. Tạo Docker network

File compose đang dùng network ngoài tên `web-net`:

```yaml
networks:
  web-net:
    external: true
```

Vì vậy cần tạo network trước khi chạy container:

```powershell
docker network create web-net
```

Kiểm tra lại:

```powershell
docker network ls
```

## 4. Khởi động các container

Di chuyển vào thư mục `settings` rồi chạy:

```powershell
cd settings
docker compose up -d
```

Nếu chỉ muốn chạy từng dịch vụ:

```powershell
docker compose up -d qdrant
docker compose up -d neo4j
```

## 5. Thông tin cấu hình từng dịch vụ

### Qdrant

- Image: `qdrant/qdrant:latest`
- Container name: `qdrant`
- REST API: `http://localhost:6333`
- gRPC: `localhost:6334`

Health check:

```powershell
curl http://localhost:6333/healthz
```

### Neo4j

- Image: `neo4j:5-community`
- Container name: `neo4j`
- Neo4j Browser: `http://localhost:7474`
- Bolt: `bolt://localhost:7687`
- Tài khoản mặc định theo compose:
  - Username: `neo4j`
  - Password: `CjeZD6XqRXhg`

Neo4j đang bật:

- Plugin `APOC`
- Heap init: `1G`
- Heap max: `2G`

Health check:

```powershell
docker ps
docker logs neo4j
```

Sau khi container chạy, có thể mở trình duyệt tại:

```text
http://localhost:7474
```

## 6. Kiểm tra trạng thái sau khi cài đặt

Kiểm tra danh sách container:

```powershell
docker ps
```

Kiểm tra log:

```powershell
docker logs qdrant
docker logs neo4j
```

Kiểm tra dịch vụ Qdrant:

```powershell
curl http://localhost:6333/healthz
```

Kiểm tra Neo4j bằng giao diện web:

- Truy cập `http://localhost:7474`
- Đăng nhập bằng tài khoản trong file compose

## 7. Dừng và khởi động lại

Dừng toàn bộ dịch vụ:

```powershell
cd settings
docker compose down
```

Khởi động lại:

```powershell
cd settings
docker compose up -d
```

Lưu ý: `docker compose down` chỉ dừng container, dữ liệu vẫn được giữ trong các thư mục volume đã map ở `settings/qdrant` và `settings/neo4j`.

## 8. Gợi ý sử dụng cho RAG

Sau khi hoàn tất:

- Đưa dữ liệu embedding vào `Qdrant` để phục vụ truy vấn semantic search.
- Đưa thực thể và quan hệ vào `Neo4j` để phục vụ graph-based retrieval.
- Ứng dụng RAG có thể kết hợp cả hai nguồn:
  - `Qdrant` để tìm đoạn văn bản gần nghĩa
  - `Neo4j` để mở rộng ngữ cảnh bằng quan hệ tri thức

## 9. Một số lỗi thường gặp

### Lỗi chưa có network `web-net`

Thông báo thường gặp:

```text
network web-net declared as external, but could not be found
```

Cách xử lý:

```powershell
docker network create web-net
```

### Lỗi port đã được sử dụng

Các port đang dùng:

- `6333`, `6334` cho Qdrant
- `7474`, `7687` cho Neo4j

Nếu bị trùng port, cần dừng dịch vụ đang chiếm port hoặc sửa lại mapping trong `settings/docker-compose.yaml`.

### Lỗi container chạy nhưng ứng dụng không kết nối được

Kiểm tra:

- Container có đang `Up` hay không: `docker ps`
- Log container có lỗi hay không: `docker logs qdrant`, `docker logs neo4j`
- Ứng dụng đang dùng đúng host/port/user/password theo file compose hay không
