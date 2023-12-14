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

use risc0_zkvm::recursion::{identity_p254, join, lift};
use risc0_zkvm::{
    get_prover_server, ExecutorEnv, ExecutorImpl, InnerReceipt, ProverOpts, Receipt,
    SegmentReceipt, VerifierContext,
};
use serde_json;
use smartcore::{
    linalg::basic::matrix::DenseMatrix, tree::decision_tree_classifier::DecisionTreeClassifier,
};
use smartcore_ml_methods::{ML_TEMPLATE_ELF, ML_TEMPLATE_ID};


// The serialized trained model and input data are embedded from files
// corresponding paths listed below. Alternatively, the model can be trained in
// the host and/or data can be manually inputted as a smartcore DenseMatrix. If
// this approach is desired, be sure to import the corresponding SmartCore
// modules and serialize the model and data to byte arrays before transfer to
// the guest.
const JSON_MODEL: &str = include_str!("../res/ml-model/tree_model_bytes.json");
const JSON_DATA: &str = include_str!("../res/input-data/tree_model_data_bytes.json");

fn main() {
    // Prove the task locally
    prove_locally();
    
    // Prove the task using client-server
    prove_server_side();
}

fn prove_locally() {
    // We set a boolean to establish whether we are using a SVM model.  This will be
    // passed to the guest and is important for execution of the guest code.
    // SVM models require an extra step that is not required of other SmartCore
    // models.
    let is_svm: bool = false;

    // Convert the model and input data from JSON into byte arrays.
    let model_bytes: Vec<u8> = serde_json::from_str(JSON_MODEL).unwrap();
    let data_bytes: Vec<u8> = serde_json::from_str(JSON_DATA).unwrap();

    // Deserialize the data from rmp into native rust types.
    type Model = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;
    let model: Model =
        rmp_serde::from_slice(&model_bytes).expect("model failed to deserialize byte array");
    let data: DenseMatrix<f64> =
        rmp_serde::from_slice(&data_bytes).expect("data filed to deserialize byte array");

    // We build the ExecutorEnv struct to pass to the guest.
    let segment_limit_po2 = 17;
    println!("segment_limit_po2 {}", segment_limit_po2);
    let env = ExecutorEnv::builder()
        .segment_limit_po2(segment_limit_po2)
        .write(&is_svm)
        .expect("bool failed to serialize")
        .write(&model)
        .expect("model failed to serialize")
        .write(&data)
        .expect("data failed to serialize")
        .build()
        .unwrap();

    // Run the executor locally to get segments.
    let mut exec = ExecutorImpl::from_elf(env, ML_TEMPLATE_ELF).unwrap();
    let session = exec.run().unwrap();
    let segments = session.resolve().unwrap();
    println!("Got {} segments.", segments.len());
    
    // Run prover locally to get succinct receipt for each segment.
    let opts = ProverOpts {
        hashfn: "poseidon".to_string(),
        prove_guest_errors: false,
    };
    let prover = get_prover_server(&opts).unwrap();
    println!("Proving rv32im");
    let ctx = VerifierContext::default();
    let segment_receipts: Vec<SegmentReceipt> = segments
        .iter()
        .map(|x| {
            let start_time: std::time::Instant = std::time::Instant::now();
            let result = prover.prove_segment(&ctx, x).unwrap();
            let elapsed = start_time.elapsed();
            println!("segment prove time = {:?}", elapsed);
            result
        })
        .collect();
    println!("Done proving rv32im");
    
    // Run recursion locally to get succinct receipt for the entire session.
    let start_time: std::time::Instant = std::time::Instant::now();
    let mut rollup: risc0_zkvm::SuccinctReceipt = lift(&segment_receipts[0]).unwrap();
    let elapsed = start_time.elapsed();
    println!("lift time = {:?}", elapsed);
    let ctx = VerifierContext::default();
    for receipt in &segment_receipts[1..] {
        let start_time: std::time::Instant = std::time::Instant::now();
        let rec_receipt = lift(receipt).unwrap();
        rec_receipt.verify_integrity_with_context(&ctx).unwrap();
        rollup = join(&rollup, &rec_receipt).unwrap();
        let elapsed = start_time.elapsed();
        rollup.verify_integrity_with_context(&ctx).unwrap();
        println!("recursion time = {:?}", elapsed);
    }
    
    // Check on stark-to-snark
    // let snark_receipt =
    identity_p254(&rollup).expect("Running prover failed");

    // Uncomment to write seal...
    // let seal: Vec<u8> =
    // bytemuck::cast_slice(snark_receipt.seal.as_slice()).into();
    // std::fs::write("recursion.seal", seal);

    // Validate the Session rollup + journal data
    // Verify that this receipt proves a successful execution of the zkVM from the given image_id
    let rollup_receipt = Receipt::new(
        InnerReceipt::Succinct(rollup),
        session.journal.unwrap().bytes,
    );
    rollup_receipt.verify(ML_TEMPLATE_ID).unwrap();
    println!("Rollup receipt verified successfully!");
    
}


use std::{
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    thread,
};

use risc0_zkvm::{Asset, AssetRequest, ConnectionWrapper, Connector, TcpConnection};
use tempfile::{tempdir, TempDir};
use risc0_zkvm::{ApiClient, SessionInfo, SuccinctReceipt, ApiServer};
use anyhow::Result;


struct TestClientConnector {
    listener: TcpListener,
}

impl TestClientConnector {
    fn new() -> Result<Self> {
        Ok(Self {
            listener: TcpListener::bind("127.0.0.1:0")?,
        })
    }
}

impl Connector for TestClientConnector {
    fn connect(&self) -> Result<ConnectionWrapper> {
        let (stream, _) = self.listener.accept()?;
        Ok(ConnectionWrapper::new(Box::new(TcpConnection::new(stream))))
    }
}

struct TestClient {
    work_dir: TempDir,
    client: ApiClient,
    addr: SocketAddr,
    segments: Vec<Asset>,
}

impl TestClient {
    fn new() -> Self {
        let connector = TestClientConnector::new().unwrap();
        let addr = connector.listener.local_addr().unwrap();
        let client = ApiClient::with_connector(Box::new(connector));
        Self {
            work_dir: tempdir().unwrap(),
            client,
            addr,
            segments: Vec::new(),
        }
    }

    fn get_work_path(&self) -> PathBuf {
        self.work_dir.path().to_path_buf()
    }

    fn execute(&mut self, env: ExecutorEnv<'_>, binary: Asset) -> SessionInfo {
        with_server(self.addr, || {
            let segments_out = AssetRequest::Path(self.get_work_path());
            self.client
                .execute(&env, binary, segments_out, |_info, asset| {
                    self.segments.push(asset);
                    Ok(())
                })
        })
    }

    // fn prove(&self, env: ExecutorEnv<'_>, opts: ProverOpts, binary: Asset) -> Receipt {
    //     with_server(self.addr, || self.client.prove(&env, opts, binary))
    // }

    fn prove_segment(&self, opts: ProverOpts, segment: Asset) -> SegmentReceipt {
        with_server(self.addr, || {
            let receipt_out = AssetRequest::Path(self.get_work_path());
            self.client.prove_segment(opts, segment, receipt_out)
        })
    }

    fn lift(&self, opts: ProverOpts, receipt: Asset) -> SuccinctReceipt {
        with_server(self.addr, || {
            let receipt_out = AssetRequest::Path(self.get_work_path());
            self.client.lift(opts, receipt, receipt_out)
        })
    }

    fn join(&self, opts: ProverOpts, left_receipt: Asset, right_receipt: Asset) -> SuccinctReceipt {
        with_server(self.addr, || {
            let receipt_out = AssetRequest::Path(self.get_work_path());
            self.client
                .join(opts, left_receipt, right_receipt, receipt_out)
        })
    }

    fn identity_p254(&self, opts: ProverOpts, receipt: Asset) -> SuccinctReceipt {
        with_server(self.addr, || {
            let receipt_out = AssetRequest::Path(self.get_work_path());
            self.client.identity_p254(opts, receipt, receipt_out)
        })
    }
}

fn with_server<T, F: FnOnce() -> Result<T>>(addr: SocketAddr, f: F) -> T {
    let addr = addr.to_string();
    let handle = thread::Builder::new()
        .name("server".into())
        .spawn(move || {
            let server = ApiServer::new_tcp(addr);
            server.run().unwrap();
        })
        .unwrap();

    let result = f().unwrap();
    handle.join().unwrap();
    result
}


fn prove_server_side() {

    let is_svm: bool = false;

    // Convert the model and input data from JSON into byte arrays.
    let model_bytes: Vec<u8> = serde_json::from_str(JSON_MODEL).unwrap();
    let data_bytes: Vec<u8> = serde_json::from_str(JSON_DATA).unwrap();

    // Deserialize the data from rmp into native rust types.
    type Model = DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>>;
    let model: Model =
        rmp_serde::from_slice(&model_bytes).expect("model failed to deserialize byte array");
    let data: DenseMatrix<f64> =
        rmp_serde::from_slice(&data_bytes).expect("data filed to deserialize byte array");

    // We build the ExecutorEnv struct to pass to the guest.
    let segment_limit_po2 = 17;
    println!("segment_limit_po2 {}", segment_limit_po2);
    let env = ExecutorEnv::builder()
        .segment_limit_po2(segment_limit_po2)
        .write(&is_svm)
        .expect("bool failed to serialize")
        .write(&model)
        .expect("model failed to serialize")
        .write(&data)
        .expect("data failed to serialize")
        .build()
        .unwrap();

    let binary = Asset::Inline(ML_TEMPLATE_ELF.into());
    let mut client: TestClient = TestClient::new();
    
    let session = client.execute(env, binary);
    assert_eq!(session.segments.len(), client.segments.len());
    let opts = ProverOpts::default();
    let receipt = client.prove_segment(opts.clone(), client.segments[0].clone());

    let mut rollup = client.lift(opts.clone(), receipt.try_into().unwrap());
    for segment in &client.segments[1..] {
        let start_time: std::time::Instant = std::time::Instant::now();
        let receipt = client.prove_segment(opts.clone(), segment.clone());
        let rec_receipt = client.lift(opts.clone(), receipt.try_into().unwrap());

        rollup = client.join(
            opts.clone(),
            rollup.try_into().unwrap(),
            rec_receipt.try_into().unwrap(),
        );
        rollup
            .verify_integrity_with_context(&VerifierContext::default())
            .unwrap();
        let elapsed = start_time.elapsed();
        println!("recursion time = {:?}", elapsed);

    }
    client.identity_p254(opts, rollup.clone().try_into().unwrap());

    let rollup_receipt = Receipt::new(InnerReceipt::Succinct(rollup), session.journal.bytes.into());
    rollup_receipt.verify(ML_TEMPLATE_ID).unwrap();

}
#[cfg(test)]
mod test {
    use risc0_zkvm::{default_executor, ExecutorEnv};
    use smartcore::{
        linalg::basic::matrix::DenseMatrix,
        svm::{
            svc::{SVCParameters, SVC},
            Kernels,
        },
    };
    use smartcore_ml_methods::ML_TEMPLATE_ELF;
    #[test]
    fn basic() {
        const EXPECTED: &[u32] = &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2,
        ];
        let result = super::predict();
        assert_eq!(EXPECTED, result);
    }
    #[test]
    fn svc() {
        // We set is_svm equal to true for a SVC model.
        let is_svm: bool = true;

        // Create sample x and y data to train a SVC.
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ]);

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        // We create the SVC params and train the SVC model.
        // The paramaters will NOT get serialized due to a serde_skip command in the
        // source code for the SVC struct.
        let knl = Kernels::linear();
        let params = &SVCParameters::default().with_c(200.0).with_kernel(knl);
        let svc = SVC::fit(&x, &y, params).unwrap();

        // This simulates importing a serialized model.
        let svc_serialized = serde_json::to_string(&svc).expect("failed to serialize");
        let svc_deserialized: SVC<f64, i32, DenseMatrix<f64>, Vec<i32>> =
            serde_json::from_str(&svc_serialized).expect("unable to deserialize JSON");

        let env = ExecutorEnv::builder()
            .write(&is_svm)
            .expect("bool failed to serialize")
            .write(&svc_deserialized)
            .expect("model failed to serialize")
            .write(&x)
            .expect("data failed to serialize")
            .build()
            .unwrap();

        // We run the executor and bypass the prover.
        let exec = default_executor();
        let session = exec.execute(env, ML_TEMPLATE_ELF).unwrap();

        // We read the result commited to the journal by the guest code.
        let result: Vec<f64> = session.journal.decode().unwrap();

        let y_expected: Vec<f64> = vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ];

        assert_eq!(result, y_expected);
    }
}
