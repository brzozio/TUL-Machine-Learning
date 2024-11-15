#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#define NUM_OF_FEATURES 5
#define NUM_OF_MOVIES 200

std::string get_repository_path(){    
    std::string repo_path = std::filesystem::current_path().generic_string();
    std::size_t found_at = repo_path.find("/src");
    if (found_at!=std::string::npos) return std::string(repo_path, 0, found_at+1);
    return "";
}

struct movie_rating{
    int m_id;
    int rating;
};

// calculates the relative distances of all pairs of movies in 'distance_coefficients'
// as a linear combination of relative distance coefficients and the provided 'metric'
// stores the result in 'scaled_dist_tensor' and sorts a matrix of movie_ids according
// to their new relative distances storing the result in 'destination_tensor'
inline void argsort_scaled_distances(int (&destiantion_tensor)[NUM_OF_MOVIES][NUM_OF_MOVIES], float (&scaled_dist_tensor)[NUM_OF_MOVIES][NUM_OF_MOVIES],
    const std::vector<std::vector<std::vector<float>>> (&distance_coefficients), const float (&metric)[NUM_OF_FEATURES]){
        
    for(int movie_1_id = 0; movie_1_id < NUM_OF_MOVIES; movie_1_id++){
        for(int feature_id = 0; feature_id < NUM_OF_FEATURES; feature_id++){
            for(int movie_2_id = 0; movie_2_id < NUM_OF_MOVIES; movie_2_id++){
                scaled_dist_tensor[movie_1_id][movie_2_id] += metric[feature_id]*distance_coefficients[movie_1_id][feature_id][movie_2_id];
            }
        }
    }

    // setting self distances to -1 to guarantee the order of arg sort 
    // in edge case of two scaled distances being 0, for example two
    // movies with same director and metric = 1 for (director-director) and 0 for others
    for(int movie_1_id = 0; movie_1_id < NUM_OF_MOVIES; movie_1_id++){
        scaled_dist_tensor[movie_1_id][movie_1_id] = -1.0;     
        for(int movie_2_id = 0; movie_2_id < NUM_OF_MOVIES; movie_2_id++){
            destiantion_tensor[movie_1_id][movie_2_id] = movie_2_id;
        }
    }

    for(int movie_id = 0; movie_id < NUM_OF_MOVIES; movie_id++){
        std::stable_sort(destiantion_tensor[movie_id], destiantion_tensor[movie_id]+NUM_OF_MOVIES,
            [&scaled_dist_tensor, &movie_id](const size_t &neigh_1_id, const size_t &neigh_2_id){ 
                return scaled_dist_tensor[movie_id][neigh_1_id] < scaled_dist_tensor[movie_id][neigh_2_id];
            });
    }

    // offsetting movie_ids to match the 1 indexed convention in the dataset
    for(int movie_1_id = 0; movie_1_id < NUM_OF_MOVIES; movie_1_id++){
        for(int movie_2_id = 0; movie_2_id < NUM_OF_MOVIES; movie_2_id++){
            destiantion_tensor[movie_1_id][movie_2_id]++;
        }
    }

}

inline int load_movie_realtive_distances(const std::string &REPO_PATH, std::vector<std::vector<std::vector<float>>> &MOVIE_DISTANCE_COEFFICIENTS_TENSOR){
    

    std::ifstream file_movie_dist_coeffs(REPO_PATH + "csv/MOVIE_DISTANCE_TREE.csv");
    if (!file_movie_dist_coeffs.is_open()) return 1;

    MOVIE_DISTANCE_COEFFICIENTS_TENSOR = std::vector<std::vector<std::vector<float>>>(
        NUM_OF_MOVIES, std::vector<std::vector<float>>(NUM_OF_FEATURES, std::vector<float>(NUM_OF_MOVIES, 0)));

    std::string line;
    
    int movie_1_id = 0, movie_2_id = 0, feature_id = 0;
    while(std::getline(file_movie_dist_coeffs,line)){
        
        MOVIE_DISTANCE_COEFFICIENTS_TENSOR[movie_1_id][feature_id][movie_2_id] = std::stof(line);

        feature_id++;
        if(feature_id >= NUM_OF_FEATURES){
            movie_2_id ++;
            feature_id = 0;
        }
        if(movie_2_id >= NUM_OF_MOVIES){
            movie_1_id ++;
            movie_2_id = 0;
        }
    }

    file_movie_dist_coeffs.close();

    return 0;
}

inline int load_user_ratings_train_data(const std::string &REPO_PATH, 
    std::unordered_map<int,int> &local_user_id, std::vector<std::vector<int>> &user_ratings, std::vector<std::vector<int>>  &user_rated_movie_id){ 

    std::ifstream file_user_ratings_train(REPO_PATH + "csv/train.csv");
    if (!file_user_ratings_train.is_open()) return 1;
    
    std::string line;
    
    int current_local_user_id = 0;
    int loaded_value = 0;
    std::vector<int> current_user_ratings;
    std::vector<int> current_user_rated_movie_id;

    // manually executing the first step to avoid uncesseraty logic inside the loop
    std::getline(file_user_ratings_train,line);
    {
        std::stringstream string_stream(line);
        std::string loaded_string;
        std::getline(string_stream, loaded_string, ';');
        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        local_user_id[loaded_value] = current_local_user_id;
        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        current_user_rated_movie_id.push_back(loaded_value);
        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        current_user_ratings.push_back(loaded_value);
    }

    while(std::getline(file_user_ratings_train,line)){
        std::stringstream string_stream(line);
        std::string loaded_string;

        // load entry ID and throw away
        std::getline(string_stream, loaded_string, ';');        

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        if(local_user_id.find(loaded_value) == local_user_id.end()){
            current_local_user_id++;
            local_user_id[loaded_value] = current_local_user_id;
            user_ratings.push_back(current_user_ratings);
            current_user_ratings.resize(0);
            user_rated_movie_id.push_back(current_user_rated_movie_id);
            current_user_rated_movie_id.resize(0);
        }

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        current_user_rated_movie_id.push_back(loaded_value);

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        current_user_ratings.push_back(loaded_value);
    }
    // pushing dangling vectors after no new key is found upon reaching EoF
    user_ratings.push_back(current_user_ratings);
    user_rated_movie_id.push_back(current_user_rated_movie_id);

    file_user_ratings_train.close();

    return 0;
}


int main(int argc, char** argv){

    const std::string REPO_PATH = get_repository_path();    
    if(REPO_PATH == "") {
        std::cout<<"UNSUPPORTED REPO STRUCTURE";
        return 1;
    }

    // indexes are swapped to capitalize on prefetching and
    // loop unrolling via parralelization of linear combinations
    std::vector<std::vector<std::vector<float>>> MOVIE_DISTANCE_COEFFICIENTS_TENSOR;

    if(load_movie_realtive_distances(REPO_PATH,MOVIE_DISTANCE_COEFFICIENTS_TENSOR)){
        std::cout<<"MOVIE RELATIVE DISTANCES COEFFICIENT DATASET NOT FOUND";
        return 2;
    }
    
    // f: datast_user_id -> local_user_id, split form the dataset to
    // reindex users for faster data fetching achieved in regular arrays
    std::unordered_map<int,int> LOCAL_USER_ID;
    std::vector<std::vector<int>> USER_RATED_MOVIE_ID;
    std::vector<std::vector<int>> USER_RATINGS;

    if(load_user_ratings_train_data(REPO_PATH, LOCAL_USER_ID, USER_RATINGS, USER_RATED_MOVIE_ID)){
        std::cout<<"USER RATINGS TRAINING DATASET NOT FOUND";
        return 3;
    }

    float movie_distance_tensor[NUM_OF_MOVIES][NUM_OF_MOVIES] = {0};
    int movie_nearest_neighbours[NUM_OF_MOVIES][NUM_OF_MOVIES] = {0};

    float metric[5] = {0.2,0.2,0.2,0.2,0.2};

    argsort_scaled_distances(movie_nearest_neighbours, movie_distance_tensor, MOVIE_DISTANCE_COEFFICIENTS_TENSOR, metric);

    return 0;
}