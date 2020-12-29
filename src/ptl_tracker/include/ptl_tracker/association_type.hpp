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
            ass_vector.push_back(ass_object);
        }
    }

    void reranking()
    {
        if (ass_vector.empty() || ass_vector.size() == 1)
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
                if (ass_obj.bbox_match_dis < iter->bbox_match_dis ||
                    iter == new_ass_vector.end() - 1)
                {
                    new_ass_vector.insert(iter, ass_obj);
                    break;
                }
            }
        }
        ass_vector = new_ass_vector;
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
    int tracker_index;
    if (detector_association_vectors[detector_index].ass_vector.empty())
        return;
    else
        // tracker only has 0 or 1 association
        tracker_index = detector_association_vectors[detector_index].ass_vector[0].id;

    ROS_INFO_STREAM("tracker index:" << tracker_index);
    ROS_INFO_STREAM("length:" << tracker_association_vectors[tracker_index].ass_vector.size());
    ROS_INFO_STREAM("local tracker size:" << tracker_association_vectors.size());
    ROS_INFO_STREAM("local detector size:" << detector_association_vectors.size());
    if (tracker_association_vectors[tracker_index].ass_vector.size() <= 1)
        return;
    else
    {
        while (tracker_association_vectors[tracker_index].ass_vector.size() > 1)
        {
            ROS_INFO("FUCK3");
            int tracker_last_detector_index = tracker_association_vectors[tracker_index].ass_vector[1].id;
            tracker_association_vectors[tracker_index].ass_vector.erase(tracker_association_vectors[tracker_index].ass_vector.begin() + 1);
            ROS_INFO("FUCK4");
            detector_association_vectors[tracker_last_detector_index].ass_vector.erase(detector_association_vectors[tracker_last_detector_index].ass_vector.begin());
            if (detector_association_vectors[tracker_last_detector_index].ass_vector.empty())
            {
                ROS_INFO("FUCK4.3");
                continue;
            }
            else
            {
                ROS_INFO("FUCK4.5");
                ROS_INFO_STREAM("tracker_last_detector_index:" << tracker_last_detector_index);
                ROS_INFO_STREAM("new size:" << detector_association_vectors[tracker_last_detector_index].ass_vector.size());
                int new_tracker_index = detector_association_vectors[tracker_last_detector_index].ass_vector.begin()->id;
                ROS_INFO("FUCK4.6");
                AssociationType new_ass = detector_association_vectors[tracker_last_detector_index].ass_vector[0];
                ROS_INFO("FUCK4.7");
                new_ass.id = tracker_last_detector_index; //change to detector id
                ROS_INFO("FUCK4.8");
                tracker_association_vectors[new_tracker_index].add_new_ass(new_ass);
                ROS_INFO("FUCK4.9");
                ROS_INFO_STREAM("tracker_last_detector_index2:" << tracker_last_detector_index);
                uniquify_detector_association_vectors_once(detector_association_vectors, tracker_association_vectors, tracker_last_detector_index);
                ROS_INFO("FUCK5.0");
            }
            ROS_INFO("FUCK5");
        }
    }
}

void uniquify_detector_association_vectors(std::vector<AssociationVector> &detector_association_vectors, int local_track_list_num)
{
    std::vector<AssociationVector> tracker_association_vectors(local_track_list_num, AssociationVector());

    //init tracker_association_vectors
    ROS_INFO("FUCK1");
    for (int i = 0; i < detector_association_vectors.size(); i++)
    {
        if (detector_association_vectors[i].ass_vector.empty())
            continue;
        else
        {
            /* code */
            ROS_INFO("FUCK1.1");

            AssociationType new_ass = detector_association_vectors[i].ass_vector[0];
            int tracker_id = new_ass.id;
            ROS_INFO_STREAM("FUCK1.1:tracker id: " << tracker_id);
            new_ass.id = i;
            tracker_association_vectors[tracker_id].add_new_ass(new_ass);
        }
    }

    for (auto t : tracker_association_vectors)
    {
        ROS_INFO_STREAM("Local tracker list: " << t.ass_vector.size());
    }

    //uniquify the vectors
    ROS_INFO("FUCK2");
    for (int i = 0; i < detector_association_vectors.size(); i++)
        uniquify_detector_association_vectors_once(detector_association_vectors, tracker_association_vectors, i);
}
