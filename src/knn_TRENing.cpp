#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <array>
#include <unordered_map>
#include <random>
#include <future>
// #include <chrono>

#define NUM_OF_FEATURES 5
#define NUM_OF_MOVIES 200

#define HYPER_PARAMS_DIM 5
#define HYPER_PARAMS_VALUES { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 }
#define HPR 6
constexpr int HYPER_PARAMS_COMBINATIONS = HPR*(HPR+1)*(HPR+2)*(HPR+3)/24;

#define TRAIN_ID_SPLIT 80
#define TRAIN_NUM_OF_MOVIES 90
#define MAX_NEIGH 6

#define NUM_OF_USERS 358

#define THREAD_COUNT 12

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
    
    std::memset(scaled_dist_tensor, 0.0, NUM_OF_MOVIES*NUM_OF_MOVIES*sizeof(float));

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
    std::vector<int> &local_user_id, std::vector<std::vector<int>> &user_movie_rating, std::vector<std::vector<int>> &user_movie_id){ 

    std::ifstream file_user_ratings_train(REPO_PATH + "csv/train.csv");
    if (!file_user_ratings_train.is_open()) return 1;
    
    std::string line;
    
    int current_local_user_id = 0;
    int loaded_value = 0;
    // binding ratings and movie ids to later shuffle for each user
    std::vector<std::vector<std::pair<int,int>>> user_ratings;
    std::vector<std::pair<int,int>> current_user_ratings;
    std::pair<int,int> pair_loaded_values;

    // manually executing the first step to avoid uncesseraty logic inside the loop
    std::getline(file_user_ratings_train,line);
    {
        std::stringstream string_stream(line);
        std::string loaded_string;

        std::getline(string_stream, loaded_string, ';');

        std::getline(string_stream, loaded_string, ';');        
        loaded_value = std::stoi(loaded_string);
        local_user_id[current_local_user_id] = loaded_value;

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        pair_loaded_values.first = loaded_value;

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        pair_loaded_values.second = loaded_value;

        current_user_ratings.push_back(pair_loaded_values);
    }

    while(std::getline(file_user_ratings_train,line)){
        std::stringstream string_stream(line);
        std::string loaded_string;

        // load entry ID and throw away
        std::getline(string_stream, loaded_string, ';');        

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        if(local_user_id[current_local_user_id] != loaded_value){

            current_local_user_id++;
            local_user_id[current_local_user_id] = loaded_value;

            user_ratings.push_back(current_user_ratings);
            current_user_ratings.resize(0);
        }

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        pair_loaded_values.first = loaded_value;

        std::getline(string_stream, loaded_string, ';');
        loaded_value = std::stoi(loaded_string);
        pair_loaded_values.second = loaded_value;

        current_user_ratings.push_back(pair_loaded_values);
    }
    // pushing dangling vectors after no new key is found upon reaching EoF
    user_ratings.push_back(current_user_ratings);
    
    file_user_ratings_train.close(); 

    std::random_device rd;
    std::mt19937 g(rd());
    
    for(int u_id = 0; u_id < NUM_OF_USERS; u_id++){
        std::shuffle(user_ratings[u_id].begin(), user_ratings[u_id].end(), g);
        for(int m_id = 0; m_id < TRAIN_NUM_OF_MOVIES; m_id++){
            user_movie_id[u_id][m_id] = user_ratings[u_id][m_id].first;
            user_movie_rating[u_id][m_id] = user_ratings[u_id][m_id].second;
        }
    }



    return 0;
}

inline void initialize_hyper_params(float (&params)[HYPER_PARAMS_COMBINATIONS][HYPER_PARAMS_DIM]){
    std::array<float,HPR> hiparams = HYPER_PARAMS_VALUES;
    
    int param_id = 0;
    for(auto x0: hiparams){for(auto x1: hiparams){for(auto x2: hiparams){for(auto x3: hiparams){for(auto x4: hiparams){
        if(x0+x1+x2+x3+x4==1){
            params[param_id][0] = x0;
            params[param_id][1] = x1;
            params[param_id][2] = x2;
            params[param_id][3] = x3;
            params[param_id][4] = x4;
            param_id++;
        }
    }}}}}
}


// "temporarily" as global variables because i'm too tierd to pretend to know
// why threads won't deploy even tough functions work corretcly when called manually

float METRIC[HYPER_PARAMS_COMBINATIONS][HYPER_PARAMS_DIM];

// f: datast_user_id -> local_user_id, split form the dataset to
// reindex users for faster data fetching achieved in regular arrays
std::vector<int> LOCAL_USER_ID(NUM_OF_USERS);
std::vector<std::vector<int>> USER_MOVIE_RATING(NUM_OF_USERS, std::vector<int>(TRAIN_NUM_OF_MOVIES));
std::vector<std::vector<int>> USER_MOVIE_ID(NUM_OF_USERS, std::vector<int>(TRAIN_NUM_OF_MOVIES));

// indexes are swapped to capitalize on prefetching and
// loop unrolling via parralelization of linear combinations
std::vector<std::vector<std::vector<float>>> MOVIE_DISTANCE_COEFFICIENTS_TENSOR;


void parallel_training_loop(const int metric_id_begin, const int metric_id_end, const void *training_results_ptr){//, const std::vector<std::vector<std::vector<float>>> &MOVIE_DISTANCE_COEFFICIENTS_TENSOR,
        // const float (&METRIC)[HYPER_PARAMS_COMBINATIONS][HYPER_PARAMS_DIM], const std::vector<std::vector<int>> &USER_MOVIE_ID,
        // const std::vector<std::vector<int>> &USER_MOVIE_RATING){

    std::vector<std::vector<float>> *training_results = (std::vector<std::vector<float>>*)training_results_ptr;

    float movie_distance_tensor[NUM_OF_MOVIES][NUM_OF_MOVIES] = {0};

    // f: "movie_id_1 - 1" -> ( g: "X'th nearest to movie_1_id" -> "movie_2_id" )
    int movie_nearest_neighbours[NUM_OF_MOVIES][NUM_OF_MOVIES] = {0};        

    // f: "movie_validation_id" -> (g: "X'th nearest" -> "rating")
    int training_nearest_neighbours_rating[TRAIN_NUM_OF_MOVIES-TRAIN_ID_SPLIT][MAX_NEIGH] = {0};

    
    int temp_sum = 0;
    int found_neighbours = 0;
    int current_gloabl_neighbour = 1;

    int current_accuracy = 0;

    for(int metric_id = metric_id_begin; metric_id < metric_id_end; metric_id++){
        
        argsort_scaled_distances(movie_nearest_neighbours, movie_distance_tensor, MOVIE_DISTANCE_COEFFICIENTS_TENSOR, METRIC[metric_id]);

        for(int u_id = 0; u_id < NUM_OF_USERS; u_id++){

            found_neighbours = 0;
            current_gloabl_neighbour = 1;
            current_accuracy = 0;

            for(int movie_valid_id = TRAIN_ID_SPLIT; movie_valid_id < TRAIN_NUM_OF_MOVIES; movie_valid_id++){
                
                while(found_neighbours < MAX_NEIGH){
                    
                    auto local_movie_id = std::find(&USER_MOVIE_ID[u_id][0], &USER_MOVIE_ID[u_id][TRAIN_ID_SPLIT], 
                            movie_nearest_neighbours[USER_MOVIE_ID[u_id][movie_valid_id]-1][current_gloabl_neighbour]);
                    
                    if(*local_movie_id != USER_MOVIE_ID[u_id][TRAIN_ID_SPLIT]){
                        training_nearest_neighbours_rating[movie_valid_id-TRAIN_ID_SPLIT][found_neighbours] = 
                            *(&USER_MOVIE_RATING[u_id][0] + (local_movie_id - &USER_MOVIE_ID[u_id][0]));
                        found_neighbours++;
                    }

                    current_gloabl_neighbour++;
                }                
                current_gloabl_neighbour=0;
                found_neighbours=0;
            }

            for(int max_nei = 2; max_nei < MAX_NEIGH; max_nei++){
                
                current_accuracy = 0;
                for(int valid_movie_id = 0; valid_movie_id < TRAIN_NUM_OF_MOVIES - TRAIN_ID_SPLIT; valid_movie_id++){
                    
                    temp_sum = 0;
                    for(int nei_count = 0; nei_count < max_nei; nei_count++){
                        temp_sum += training_nearest_neighbours_rating[valid_movie_id][nei_count];
                    }

                    if(int(round(temp_sum/max_nei)) == USER_MOVIE_RATING[u_id][valid_movie_id+TRAIN_ID_SPLIT]) current_accuracy++;
                    
                }
                
                if(current_accuracy > (*training_results)[u_id][0]){
                    (*training_results)[u_id][0] = current_accuracy;
                    (*training_results)[u_id][1] = metric_id;
                    (*training_results)[u_id][2] = max_nei;
                }
            }
        }
    }
}

int main(int argc, char** argv){

    const std::string REPO_PATH = get_repository_path();    
    if(REPO_PATH == "") {
        std::cout<<"UNSUPPORTED REPO STRUCTURE";
        return 1;
    }


    if(load_movie_realtive_distances(REPO_PATH, MOVIE_DISTANCE_COEFFICIENTS_TENSOR)){
        std::cout<<"MOVIE RELATIVE DISTANCES COEFFICIENT DATASET NOT FOUND";
        return 2;
    }
    

    if(load_user_ratings_train_data(REPO_PATH, LOCAL_USER_ID, USER_MOVIE_RATING, USER_MOVIE_ID)){
        std::cout<<"USER RATINGS TRAINING DATASET NOT FOUND";
        return 3;
    }

    initialize_hyper_params(METRIC);

    // f: "thread_id" -> ( g: "local_u_id" -> ( h: {0, 1, 2} -> {"accuracy", "metric_id", "neighbour_count"} ))
    std::vector<std::vector<std::vector<float>>> training_results(
        THREAD_COUNT, std::vector<std::vector<float>>(NUM_OF_USERS, std::vector<float>(3)));

    std::vector<std::thread> worker_threads;

    constexpr int thread_workload = HYPER_PARAMS_COMBINATIONS / THREAD_COUNT;

    // auto start = std::chrono::high_resolution_clock::now();

    for(int thread_id = 0; thread_id < (THREAD_COUNT-1); thread_id ++){
        worker_threads.push_back(std::thread(parallel_training_loop, thread_id*thread_workload, (thread_id+1)*thread_workload, &training_results[thread_id]));
    }
    worker_threads.push_back(std::thread(parallel_training_loop, (THREAD_COUNT-1)*thread_workload, HYPER_PARAMS_COMBINATIONS, &training_results[THREAD_COUNT-1]));

    for(auto &threads: worker_threads){
        threads.join();
    }

    // auto end = std::chrono::high_resolution_clock::now();

    for(auto &worker: training_results){
        for(int user = 140; user < 150; user++){
            std::cout<<"uid: "<<user<<",\tacc: "<<worker[user][0]<<"\n";
        }
        std::cout<<"\nnext worker\n";
    }

    // std::cout<<"\nTraining Loop Execution Time:\t"<<end-start<<"\n\n";

    return 0;
}