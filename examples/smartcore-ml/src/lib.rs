// Copyright 2023 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;
use std::fs::File;
// use itertools::Itertools;
use std::io::Write;
// use std::io::Read;
use smartcore::{
    linalg::basic::matrix::DenseMatrix, tree::decision_tree_classifier::DecisionTreeClassifier,
};
use smartcore_ml_methods::ML_TEMPLATE_ELF;

use smartcore::readers;

// use polars::prelude::*;
use serde_json;
use rmp_serde;

// use tokio::{
//     net::TcpStream,
//     sync::{
//         mpsc,
//         mpsc::{Receiver, Sender},
//         Mutex,
//     },
//     task,
//     time::{sleep, timeout},
// };
// use tokio::time::Duration;

// The serialized trained model and input data are embedded from files
// corresponding paths listed below. Alternatively, the model can be trained in
// the host and/or data can be manually inputted as a smartcore DenseMatrix. If
// this approach is desired, be sure to import the corresponding SmartCore
// modules and serialize the model and data to byte arrays before transfer to
// the guest.
const JSON_MODEL: &str = include_str!("../res/ml-model/tree_model_bytes.json");
const JSON_DATA: &str = include_str!("../res/input-data/tree_model_data_bytes.json");


// fn predict() -> Vec<u32> {
//     // We set a boolean to establish whether we are using a SVM model.  This will be
//     // passed to the guest and is important for execution of the guest code.
//     // SVM models require an extra step that is not required of other SmartCore
//     // models.
//     let is_svm: bool = false;

//     // Convert the model and input data from JSON into byte arrays.
//     let model_bytes: Vec<u8> = serde_json::from_str(JSON_MODEL).unwrap();
//     let data_bytes: Vec<u8> = serde_json::from_str(JSON_DATA).unwrap();

//     // Deserialize the data from rmp into native rust types.
//     type Model = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;
//     let model: Model =
//         rmp_serde::from_slice(&model_bytes).expect("model failed to deserialize byte array");
//     let data: DenseMatrix<f64> =
//         rmp_serde::from_slice(&data_bytes).expect("data filed to deserialize byte array");

//     let env = ExecutorEnv::builder()
//         .write(&is_svm)
//         .expect("bool failed to serialize")
//         .write(&model)
//         .expect("model failed to serialize")
//         .write(&data)
//         .expect("data failed to serialize")
//         .build()
//         .unwrap();

//     // Obtain the default prover.
//     // Note that for development purposes we do not need to run the prover. To
//     // bypass the prover, use:
//     // ```
//     // RISC0_DEV_MODE=1 cargo run -r
//     // ```
//     // let time_started = Instant::now();

//     let prover = default_prover();

//     // This initiates a session, runs the STARK prover on the resulting exection
//     // trace, and produces a receipt.
//     let receipt = prover.prove_elf(env, ML_TEMPLATE_ELF).unwrap();

//     // We read the result that the guest code committed to the journal. The
//     // receipt can also be serialized and sent to a verifier.
//     receipt.journal.decode().unwrap()
// }

pub async fn generate_proof(input:String) -> String{

    // tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    // We set a boolean to establish whether we are using a SVM model.  This will be
    // passed to the guest and is important for execution of the guest code.
    // SVM models require an extra step that is not required of other SmartCore
    // models.
    let is_svm: bool = false;

    //write the input string to input_data.csv
    let mut file = std::fs::File::create("input_data.csv").expect("create failed");
    file.write_all("sepal.length,sepal.width,petal.length,petal.width".as_bytes()).expect("write failed");
    let input_string= format!("{}{}","\n",input);
    file.write_all(input_string.as_bytes()).expect("write failed");
    println!("data written to file" );

    let input = readers::csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
        File::open("input_data.csv").unwrap(),
        readers::csv::CSVDefinition::default()
    ).unwrap();

    let data_bytes = rmp_serde::to_vec(&input).unwrap();
    let x_json = serde_json::to_string(&data_bytes).unwrap();
    let mut f1 = File::create("./tree_model_data_bytes.json").expect("unable to create file");
    f1.write_all(x_json.as_bytes()).expect("Unable to write data");


    let data = std::fs::read_to_string("./tree_model_data_bytes.json").unwrap();

    // Convert the model and input data from JSON into byte arrays.
    let model_bytes: Vec<u8> = serde_json::from_str(JSON_MODEL).unwrap();
    let data_bytes: Vec<u8> = serde_json::from_str(&data).unwrap();

    // Deserialize the data from rmp into native rust types.
    type Model = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;
    let model: Model =
        rmp_serde::from_slice(&model_bytes).expect("model failed to deserialize byte array");
    let data: DenseMatrix<f64> =
        rmp_serde::from_slice(&data_bytes).expect("data filed to deserialize byte array");

    let env = ExecutorEnv::builder()
        .write(&is_svm)
        .expect("bool failed to serialize")
        .write(&model)
        .expect("model failed to serialize")
        .write(&data)
        .expect("data failed to serialize")
        .build()
        .unwrap();

    // Obtain the default prover.
    // Note that for development purposes we do not need to run the prover. To
    // bypass the prover, use:
    // ```
    // RISC0_DEV_MODE=1 cargo run -r
    // ```

    let time_started = Instant::now();

    let prover = default_prover();

    // This initiates a session, runs the STARK prover on the resulting exection
    // trace, and produces a receipt.
    let receipt = prover.prove_elf(env, ML_TEMPLATE_ELF).unwrap();

    println!("this demo task time consume:{:?}",Instant::now().duration_since(time_started).as_secs() as u32);
    //receipt.journal.decode().unwrap()

    // We read the result that the guest code committed to the journal. The
    // receipt can also be serialized and sent to a verifier.
    let result:Vec<u32>= receipt.journal.decode().unwrap();
    // let res = Itertools::join(&mut result.iter(), ",");
    if result.len()==1 {
        return result[0].to_string()
    }
    return "-1".to_string()
}


pub fn generate_proof_test(input:String) -> String{

    // tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    // We set a boolean to establish whether we are using a SVM model.  This will be
    // passed to the guest and is important for execution of the guest code.
    // SVM models require an extra step that is not required of other SmartCore
    // models.
    let is_svm: bool = false;

    //write the input string to input_data.csv
    let mut file = std::fs::File::create("input_data.csv").expect("create failed");
    file.write_all("sepal.length,sepal.width,petal.length,petal.width".as_bytes()).expect("write failed");
    let input_string= format!("{}{}","\n",input);
    file.write_all(input_string.as_bytes()).expect("write failed");
    println!("data written to file" );

    let input = readers::csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
        File::open("input_data.csv").unwrap(),
        readers::csv::CSVDefinition::default()
    ).unwrap();

    let data_bytes = rmp_serde::to_vec(&input).unwrap();
    let x_json = serde_json::to_string(&data_bytes).unwrap();
    let mut f1 = File::create("res/input-data/tree_model_data_bytes.json").expect("unable to create file");
    f1.write_all(x_json.as_bytes()).expect("Unable to write data");


    // Convert the model and input data from JSON into byte arrays.
    let model_bytes: Vec<u8> = serde_json::from_str(JSON_MODEL).unwrap();
    let data_bytes: Vec<u8> = serde_json::from_str(JSON_DATA).unwrap();

    // Deserialize the data from rmp into native rust types.
    type Model = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;
    let model: Model =
        rmp_serde::from_slice(&model_bytes).expect("model failed to deserialize byte array");
    let data: DenseMatrix<f64> =
        rmp_serde::from_slice(&data_bytes).expect("data filed to deserialize byte array");

    let env = ExecutorEnv::builder()
        .write(&is_svm)
        .expect("bool failed to serialize")
        .write(&model)
        .expect("model failed to serialize")
        .write(&data)
        .expect("data failed to serialize")
        .build()
        .unwrap();

    // Obtain the default prover.
    // Note that for development purposes we do not need to run the prover. To
    // bypass the prover, use:
    // ```
    // RISC0_DEV_MODE=1 cargo run -r
    // ```

    let time_started = Instant::now();

    let prover = default_prover();

    // This initiates a session, runs the STARK prover on the resulting exection
    // trace, and produces a receipt.
    let receipt = prover.prove_elf(env, ML_TEMPLATE_ELF).unwrap();

    println!("this demo task time consume:{:?}",Instant::now().duration_since(time_started).as_secs() as u32);
    //receipt.journal.decode().unwrap()

    // We read the result that the guest code committed to the journal. The
    // receipt can also be serialized and sent to a verifier.
    let result:Vec<u32>= receipt.journal.decode().unwrap();
    if result.len()==1 {
        return result[0].to_string()
    }
    return "-1".to_string()
}
