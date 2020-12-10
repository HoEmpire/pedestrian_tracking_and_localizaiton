import torch
import numpy as np
import rospy
from inference import cal_dis
from reid_config import Config


class Object():
    def __init__(self, id, feat):
        self.id = id
        self.feats = feat

    def add_feat(self, feat):
        feat.feats = torch.cat((self.feats, feat), 0)

    def is_full(self):
        if self.feats.shape[0] > 20:
            return True
        else:
            return False


class ReIDDatabase():
    def __init__(self):
        self.object_list = []
        self.feat_id_list = []
        self.feat_all = torch.tensor([]).cuda()
        self.object_num = 0
        self.cfg = Config()

    def init_database(self, feat):
        self.add_new_object(feat.unsqueeze(0))

    def add_new_object(self, feat):
        new_object = Object(self.object_num, feat)
        self.object_list.append(new_object)
        self.feat_id_list.append(new_object.id)
        if self.feat_all.shape[0] == 0:
            self.feat_all = feat
        else:
            self.feat_all = torch.cat((self.feat_all, feat), 0)
        rospy.loginfo("adding new object with id:%d", self.object_num)
        self.object_num = self.object_num + 1
        return self.object_num - 1

    def add_new_feat(self, feats, id):
        for f in feats:
            distmat = cal_dis(f.unsqueeze(0), self.object_list[id].feats)
            rank = np.argsort(distmat, axis=1).squeeze(0)
            # rospy.loginfo("In add new features of id: %d", id)
            # rospy.loginfo(distmat[0][rank])
            if (self.cfg.similarity_test_threshold < distmat[0][rank[0]] <
                    self.cfg.same_id_threshold) & (
                        not self.object_list[id].is_full()):
                rospy.loginfo("adding new feature to id:%d", id)
                # update global database
                self.feat_all = torch.cat((self.feat_all, f.unsqueeze(0)), 0)
                self.feat_id_list.append(id)

                #update local database
                self.object_list[id].feats = torch.cat(
                    (self.object_list[id].feats, f.unsqueeze(0)), 0)