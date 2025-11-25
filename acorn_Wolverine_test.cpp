#include "hnsw_Wolverine/acornlib.h"
#include "hnsw_Wolverine/acorn_Wolverine.h"
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>

using namespace std;

double average(double arr[], int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum / size;
}

bool predicate_1(const acornlib::labeltype label) {
    return label < 80000;
}

int main(int argc, char* argv[]) {
    int32_t dim = 0;               // Dimension of the elements
    int32_t max_elements = 0;   // Maximum number of elements, should be known beforehand
    float* data=nullptr;
    int32_t query_sum=0;
    int32_t query_dim=0;
    float* querys=nullptr;
    int32_t groundtruth_sum=0;
    int32_t groundtruth_dim=0;
    uint32_t* groundtruth=nullptr;

    int circul_sum=100;
    float delete_parts=0.05;
    int delete_model=0;

    string data_file_path="./dataset/sift_learn.fbin";
    string query_file_path="./dataset/sift_query.fbin";
    string groundtruth_file_path="./dataset/sift_query_learn_gt100";
    string search_result_file_path="./search_result";
    string index_prefix="./index/hnsw";
    int K=10;
    vector<size_t>deleteList;
    int M=16;
    int ef=200;
    int num_threads = 64;
    int newLinkSize=M;
    string deleteModelName[]={"VIOLENT_DELETE","PINTOPOUT_DELETE","SEARCH_DELETE","TWOHOP_DELETE","APPROXIMATE_TWOHOP_DELETE","REFACTOR_DELETE"};

    if(argc!=13&&argc!=1){
        cout<<"Missing parameters!!!!!!!!!!!!!!  "<<argc<<endl;
        throw;
    }   
    if(argc>1){
        M=atoi(argv[1]);
        ef=atoi(argv[2]);
        newLinkSize=atoi(argv[3]);
        num_threads=atoi(argv[4]);
        circul_sum=atoi(argv[5]);
        delete_parts=atof(argv[6]);
        delete_model=atoi(argv[7]);
        data_file_path.clear();
        query_file_path.clear();
        index_prefix.clear();
        groundtruth_file_path.clear();
        search_result_file_path.clear();
        data_file_path+=argv[8];
        query_file_path+=argv[9];
        groundtruth_file_path+=argv[10];
        search_result_file_path+=argv[11];
        index_prefix+=argv[12];
    }
    cout << "-----------------------------------------------------------------------------------" << endl;
    cout << "M: " << M
         << " ef: " << ef 
         << " newLinkSize: " << newLinkSize
         << " num_threads: " << num_threads
         << " circul_sum: " << circul_sum
         << " delete_parts: " << delete_parts
         << " delete_model: " << deleteModelName[delete_model] << endl;
    cout << "data_file_path: " << data_file_path 
         << " query_file_path: " << query_file_path << endl
         << " groundtruth_file_path: " << groundtruth_file_path 
         << " search_result_file_path: " << search_result_file_path
         << " index_prefix: " << index_prefix << endl;
    cout << "-----------------------------------------------------------------------------------" << endl;

    readInitData<float>(dim,max_elements,data,data_file_path);
    readQuerys<float>(query_sum,query_dim,querys,query_file_path);
    readGroundTruth<float>(groundtruth_sum,groundtruth_dim,groundtruth,groundtruth_file_path,K, predicate_1);

    cout << "max_elements: " << max_elements << " dim: " << dim << endl;
    cout << "query_sum: " << query_sum << " query_dim: " << query_dim << endl;
    cout << "groundtruth_sum: " << groundtruth_sum << " groundtruth_dim: " << groundtruth_dim << endl;
    cout << "Initing index" << endl;

    std::ofstream result_writer(search_result_file_path);
    result_writer << "recall,search_OPS,delete_OPS,insert_OPS" << endl;

    unsigned int seed = 100;
    default_random_engine random(seed);
    uniform_int_distribution<size_t> dis1(0, max_elements-1);

    acornlib::ACORN<float>* alg_acorn = nullptr;
    hnswlib::L2Space space(dim);
    creat_index(alg_acorn,index_prefix,&space,M,ef,dim,max_elements,data,num_threads);

    double* recalls = (double*) malloc(circul_sum * sizeof(double));
    double* search_OPSs = (double*) malloc(circul_sum * sizeof(double));
    double* delete_OPSs = (double*) malloc(circul_sum * sizeof(double));
    double* add_OPSs = (double*) malloc(circul_sum * sizeof(double));

    pair<float,double> search_result;
    
    #ifdef PSEDO
    cout << endl << "psedo_deletion!!!!!!!!!!!" << endl;
    alg_acorn->resizeIndex(10000000);
    #endif
    
    cout << "maxlevel_: " << alg_acorn->maxlevel_ << endl;
    cout << "start search" << endl;

    search_result = search_index(alg_acorn,
                                 K,
                                 query_sum,
                                 query_dim,
                                 querys,
                                 groundtruth_sum,
                                 groundtruth_dim,
                                 groundtruth,
                                 num_threads,
                                 predicate_1);
    for(int cir_times=0; cir_times<circul_sum; cir_times++){
        search_result = search_index(alg_acorn,
                                     K,
                                     query_sum,
                                     query_dim,
                                     querys,
                                     groundtruth_sum,
                                     groundtruth_dim,
                                     groundtruth,
                                     num_threads,
                                     predicate_1);
        cout << "cir_times: " << cir_times << '\t' 
             << "recall: " << search_result.first << '\t' 
             << "search_OPS: " << search_result.second << " query\\second\t";
        result_writer << search_result.first << ',' << search_result.second << ',';

        creat_deleteList(deleteList,max_elements*delete_parts*2,max_elements*delete_parts,random,dis1);
        write_Vector(deleteList,"deleteList");
        #ifdef PSEDO
        double deleteTime=psedo_deleteIndex(alg_acorn,deleteList,delete_model,num_threads,newLinkSize);
        #else
        double deleteTime=deleteIndex(alg_acorn,deleteList,delete_model,num_threads,newLinkSize);
        #endif
        cout<<"delete_ops: "<<deleteTime<<" point\\second\t";
        result_writer<<deleteTime<<',';

        double addTime_avg=addPoint(alg_acorn,deleteList,num_threads,data,dim);
        cout<<"add_ops: "<<addTime_avg<<" point\\second"<<endl;
        result_writer<<addTime_avg<<endl;

        recalls[cir_times] = search_result.first;
        search_OPSs[cir_times] = search_result.second;
        delete_OPSs[cir_times] = deleteTime;
        add_OPSs[cir_times] = addTime_avg;
    }
    result_writer.close();

    cout << endl;
    cout << "Average Recall:      " << average(recalls, circul_sum) << endl;
    cout << "Average Search OPS:  " << average(search_OPSs, circul_sum) << " query / second" << endl;
    cout << "Average Delete OPS:  " << average(delete_OPSs, circul_sum) << " point / second" << endl;
    cout << "Average Add OPS:     " << average(add_OPSs, circul_sum) << " point / second" << endl;
    cout << endl;

    delete[] data;
    delete[] querys;
    delete[] groundtruth;
    delete alg_acorn;

    delete[] recalls;
    delete[] search_OPSs;
    delete[] delete_OPSs;
    delete[] add_OPSs;

    data=nullptr;
    querys=nullptr;
    groundtruth=nullptr;
    alg_acorn=nullptr;

    return 0;
}