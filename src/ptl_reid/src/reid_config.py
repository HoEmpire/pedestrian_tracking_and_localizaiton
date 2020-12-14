import yaml
import rospkg


class Config():
    def __init__(self):
        rospack = rospkg.RosPack()
        package_root = rospack.get_path("ptl_reid")
        f = open(package_root + "/config/config.yaml", encoding='utf-8')
        data = yaml.load(f)
        self.similarity_test_threshold = data['similarity_test_threshold']
        self.same_id_threshold = data['same_id_threshold']
        self.batch_ratio = data['batch_ratio']
        self.object_img_num = data['object_img_num']
        self.weights_path = data['weights_path']
        self.query_batch_size = data['query_batch_size']
        f.close()


if __name__ == "__main__":
    cfg = Config()
    print(cfg.similarity_test_threshold)
    print(cfg.same_id_threshold)
    print(cfg.batch_ratio)
    print(cfg.object_img_num)
    print(cfg.weights_path)
    print(cfg.query_batch_size)