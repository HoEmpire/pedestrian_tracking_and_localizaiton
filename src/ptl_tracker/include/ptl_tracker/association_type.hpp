#pragma once
#include <ros/ros.h>

class AssociationType
{
public:
    AssociationType(int id_, float score_, float bbox_match_dis_)
    {
        id = id_;
        score = score_;
        bbox_match_dis = bbox_match_dis_;
    }
    int id;
    float score;
    float bbox_match_dis;
};

class AssociationVector
{
public:
    AssociationVector() = default;

    void add_new_ass(AssociationType ass_object)
    {
        if (ass_vector.empty())
            ass_vector.push_back(ass_object);
        else
        {
            for (auto iter = ass_vector.begin(); iter < ass_vector.end(); iter++)
            {
                if (ass_object.score < iter->score)
                {
                    ass_vector.insert(iter, ass_object);
                    return;
                }
            }
        }
    }

    void reranking()
    {
        if (ass_vector.empty())
            return;
        std::vector<AssociationType> new_ass_vector;
        for (auto ass_obj : ass_vector)
        {
            if (new_ass_vector.empty())
            {
                new_ass_vector.push_back(ass_obj);
                continue;
            }
            for (auto iter = new_ass_vector.begin(); iter < new_ass_vector.end(); iter++)
            {
                if (ass_obj.bbox_match_dis < iter->bbox_match_dis)
                {
                    ass_vector.insert(iter, ass_obj);
                    break;
                }
            }
        }
    }

    void report()
    {
        ROS_INFO("Data Association Report:");
        for (auto ass : ass_vector)
            ROS_INFO_STREAM("id: " << ass.id << " | score: " << ass.score);
    }

    std::vector<AssociationType> ass_vector;
};

void uniquify_detector_association_vectors_once(std::vector<AssociationVector> &detector_association_vectors, std::vector<AssociationVector> &tracker_association_vectors, int detector_index)
{
    // tracker only has 0 or 1 association
    int tracker_index = detector_association_vectors[detector_index].ass_vector[0].id;
    if (tracker_association_vectors[tracker_index].ass_vector.size() <= 1)
        return;
    else
    {
        while (tracker_association_vectors[tracker_index].ass_vector.size() >= 1)
        {
            int tracker_last_detector_index = tracker_association_vectors[tracker_index].ass_vector.end()->id;
            tracker_association_vectors.pop_back();

            //remove the first item
            detector_association_vectors[tracker_last_detector_index].ass_vector.erase(detector_association_vectors[tracker_last_detector_index].ass_vector.begin());
            if (detector_association_vectors[tracker_last_detector_index].ass_vector.empty())
                continue;
            else
            {
                int new_tracker_index = detector_association_vectors[tracker_last_detector_index].ass_vector.begin()->id;
                AssociationType new_ass = detector_association_vectors[tracker_last_detector_index].ass_vector[0];
                new_ass.id = tracker_last_detector_index; //change to detector id
                tracker_association_vectors[new_tracker_index].add_new_ass(new_ass);
                uniquify_detector_association_vectors_once(detector_association_vectors, tracker_association_vectors, tracker_last_detector_index);
            }
        }
    }
}

void uniquify_detector_association_vectors(std::vector<AssociationVector> detector_association_vectors, int local_track_list_num)
{
    std::vector<AssociationVector> tracker_association_vectors(local_track_list_num, AssociationVector());

    //init tracker_association_vectors
    for (int i = 0; i < detector_association_vectors.size(); i++)
    {
        AssociationType new_ass = detector_association_vectors[i].ass_vector[0];
        int tracker_id = new_ass.id;
        new_ass.id = i;
        tracker_association_vectors[tracker_id].add_new_ass(new_ass);
    }

    //uniquify the vectors
    for (int i = 0; i < detector_association_vectors.size(); i++)
        uniquify_detector_association_vectors_once(detector_association_vectors, tracker_association_vectors, i);
}
