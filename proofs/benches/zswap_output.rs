//! Bechmarks for the prover and verifier performance on the Zswap-output
//! circuit from the zswap protocol.
//!
//! For more details, visit:
//! https://github.com/midnightntwrk/midnight-ledger-prototype/blob/main/zswap/zswap.compact
//!
//! `ZSwap Prover` and `ZSwap Verifier` retain the legacy phase benchmarks.
//! `ZSwap Prover End-to-end/k=…` measures the normal public prover path under
//! the policy selected by `MIDNIGHT_SPILL_PK`, `MIDNIGHT_SPILL_COSETS`, and
//! `MIDNIGHT_SPILL_FLOOR_K`. Run each policy in a separate process and use
//! Criterion's `--save-baseline`/`--baseline` options for comparisons.
//!
//! The separately printed first-proof observation begins after the SRS and
//! proving key are ready. It is not a process or container cold-start metric.
use std::{hint::black_box, time::Instant};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use ff::Field;
use group::Group;
use midnight_circuits::{
    ecc::{hash_to_curve::HashToCurveGadget, native::EccChip},
    hash::poseidon::PoseidonChip,
    instructions::{
        AssignmentInstructions, ConversionInstructions, DecompositionInstructions, EccInstructions,
        HashToCurveCPU, PublicInputInstructions,
    },
    types::{AssignedBit, AssignedByte, AssignedNative, AssignedNativePoint, Instantiable},
};
use midnight_curves::{Bls12, Fr as JubjubScalar, JubjubExtended as Jubjub, JubjubSubgroup};
use midnight_proofs::{
    circuit::{Layouter, Value},
    plonk::{
        bench::prover::benchmark_create_proof, create_proof, keygen_pk, keygen_vk_with_k,
        parse_trace, verify_algebraic_constraints, Error,
    },
    poly::{
        commitment::Guard,
        kzg::{params::ParamsKZG, KZGCommitmentScheme},
    },
    transcript::{CircuitTranscript, Transcript},
};
use midnight_zk_stdlib::{MidnightCircuit, Relation, ZkStdLib, ZkStdLibArch};
use rand::{rngs::OsRng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::Digest;

type F = midnight_curves::Fq;
type C = midnight_curves::G1Projective;

type CoinCom = [u8; 32];
type ValueCom = JubjubSubgroup;

#[derive(Clone, Copy, Debug)]
pub enum PK {
    ZSwapCoinPublicKey([u8; 32]),
    ContractAddress([u8; 32]),
}

#[derive(Clone, Copy, Debug)]
pub struct CoinInfo {
    color: F,
    nonce: F,
    value: u64,
}

#[derive(Clone, Debug)]
struct AssignedPK {
    bytes: [AssignedByte<F>; 32],
    is_contract: AssignedBit<F>,
}

#[derive(Clone, Debug)]
struct AssignedCoinInfo {
    color_bytes: [AssignedByte<F>; 32],
    nonce_bytes: [AssignedByte<F>; 32],
    value_bytes: [AssignedByte<F>; 8],

    color: AssignedNative<F>,
    value: AssignedNative<F>,
}

#[derive(Clone, Default)]
pub struct ZSwapOutputCircuit;

impl Relation for ZSwapOutputCircuit {
    type Instance = (CoinCom, ValueCom);

    type Witness = (PK, CoinInfo, JubjubScalar);

    type Error = Error;

    fn format_instance(instance: &Self::Instance) -> Result<Vec<F>, Error> {
        let mut pi: Vec<F> =
            instance.0.iter().flat_map(AssignedByte::<F>::as_public_input).collect();
        pi.extend(AssignedNativePoint::<Jubjub>::as_public_input(&instance.1));
        Ok(pi)
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<F>,
        _instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        let pk = assign_pk(std_lib, layouter, witness.as_ref().map(|w| w.0))?;
        let coin = assign_coin(std_lib, layouter, witness.as_ref().map(|w| w.1))?;
        let domain_sep = assign_fixed_domain_sep(std_lib, layouter, "mdn:cc")?;

        let coin_com = {
            std_lib.sha2_256(
                layouter,
                &[
                    coin.color_bytes.to_vec(),
                    coin.nonce_bytes.to_vec(),
                    coin.value_bytes.to_vec(),
                    vec![pk.is_contract.into()],
                    pk.bytes.to_vec(),
                    domain_sep,
                ]
                .concat(),
            )?
        };

        let value_com = {
            let color_base = std_lib.hash_to_curve(layouter, &[coin.color])?;
            let gen = std_lib.jubjub().assign_fixed(layouter, JubjubSubgroup::generator())?;
            let rc = std_lib.jubjub().assign(layouter, witness.as_ref().map(|w| w.2))?;
            let coin_value_as_scalar = std_lib.jubjub().convert(layouter, &coin.value)?;
            std_lib
                .jubjub()
                .msm(layouter, &[coin_value_as_scalar, rc], &[color_base, gen])?
        };

        coin_com
            .iter()
            .try_for_each(|b| std_lib.constrain_as_public_input(layouter, b))?;

        std_lib.jubjub().constrain_as_public_input(layouter, &value_com)
    }

    fn write_relation<W: std::io::Write>(&self, _writer: &mut W) -> std::io::Result<()> {
        Ok(())
    }

    fn read_relation<R: std::io::Read>(_reader: &mut R) -> std::io::Result<Self> {
        Ok(ZSwapOutputCircuit)
    }

    fn used_chips(&self) -> ZkStdLibArch {
        ZkStdLibArch {
            jubjub: true,
            sha2_256: true,
            poseidon: true,
            ..ZkStdLibArch::default()
        }
    }
}

fn assign_pk(
    std_lib: &ZkStdLib,
    layouter: &mut impl Layouter<F>,
    pk: Value<PK>,
) -> Result<AssignedPK, Error> {
    let (bytes_val, is_contract_val) = pk
        .map(|pk| match pk {
            PK::ZSwapCoinPublicKey(bytes) => (bytes, false),
            PK::ContractAddress(bytes) => (bytes, true),
        })
        .unzip();

    let bytes = std_lib.assign_many(layouter, &bytes_val.transpose_array())?;

    Ok(AssignedPK {
        bytes: bytes.try_into().unwrap(),
        is_contract: std_lib.assign(layouter, is_contract_val)?,
    })
}

fn assign_coin(
    std_lib: &ZkStdLib,
    layouter: &mut impl Layouter<F>,
    coin: Value<CoinInfo>,
) -> Result<AssignedCoinInfo, Error> {
    let color = std_lib.assign(layouter, coin.map(|coin| coin.color))?;
    let nonce = std_lib.assign(layouter, coin.map(|coin| coin.nonce))?;
    let value = std_lib.assign(layouter, coin.map(|coin| F::from(coin.value)))?;

    let color_bytes = std_lib.assigned_to_le_bytes(layouter, &color, Some(32))?;
    let nonce_bytes = std_lib.assigned_to_le_bytes(layouter, &nonce, Some(32))?;
    let value_bytes = std_lib.assigned_to_le_bytes(layouter, &value, Some(8))?;

    Ok(AssignedCoinInfo {
        color_bytes: color_bytes.try_into().unwrap(),
        nonce_bytes: nonce_bytes.try_into().unwrap(),
        value_bytes: value_bytes.try_into().unwrap(),
        color,
        value,
    })
}

fn assign_fixed_domain_sep(
    std_lib: &ZkStdLib,
    layouter: &mut impl Layouter<F>,
    domain_sep: &str,
) -> Result<Vec<AssignedByte<F>>, Error> {
    std_lib.assign_many_fixed(layouter, domain_sep.as_bytes())
}

fn sample_zswap_inputs(k: u32) -> (Vec<F>, MidnightCircuit<'static, ZSwapOutputCircuit>) {
    let mut rng = ChaCha8Rng::from_entropy();

    let zswap_pk_bytes = core::array::from_fn(|_| rng.gen());
    let zswap_pk_is_contract: bool = rng.gen();
    let zswap_pk = match zswap_pk_is_contract {
        false => PK::ZSwapCoinPublicKey(zswap_pk_bytes),
        true => PK::ContractAddress(zswap_pk_bytes),
    };

    let coin = CoinInfo {
        color: F::from(42),
        nonce: F::random(&mut rng),
        value: 1000,
    };

    let coin_com: [u8; 32] = {
        let mut preimage = coin.color.to_bytes_le().to_vec();
        preimage.extend(coin.nonce.to_bytes_le().to_vec());
        preimage.extend(coin.value.to_le_bytes().to_vec());
        preimage.push(zswap_pk_is_contract as u8);
        preimage.extend(zswap_pk_bytes.map(|b| b).to_vec());
        preimage.extend("mdn:cc".as_bytes());
        sha2::Sha256::digest(preimage).into()
    };

    let rc = JubjubScalar::random(rng);
    let value_com = {
        let coin_base = <HashToCurveGadget<
            F,
            Jubjub,
            AssignedNative<F>,
            PoseidonChip<F>,
            EccChip<Jubjub>,
        > as HashToCurveCPU<Jubjub, F>>::hash_to_curve(&[coin.color]);
        JubjubSubgroup::generator() * rc + coin_base * JubjubScalar::from(coin.value)
    };

    let witness = (zswap_pk, coin, rc);
    let instance = (coin_com, value_com);

    let circuit = MidnightCircuit::new(
        &ZSwapOutputCircuit,
        Value::known(instance),
        Value::known(witness),
        Some(k),
    );

    (
        ZSwapOutputCircuit::format_instance(&instance).unwrap(),
        circuit,
    )
}

fn flag_enabled(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true"))
}

fn benchmark_memory_profile(k: u32) -> (&'static str, bool) {
    let spill_pk = flag_enabled("MIDNIGHT_SPILL_PK");
    let spill_floor = std::env::var("MIDNIGHT_SPILL_FLOOR_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(18);
    let spill_cosets = flag_enabled("MIDNIGHT_SPILL_COSETS") && k >= spill_floor;

    assert!(
        cfg!(feature = "disk-spill") || !(spill_pk || spill_cosets),
        "spill flags require the midnight-proofs/disk-spill feature"
    );

    let name = match (spill_pk, spill_cosets) {
        (false, false) => "heap",
        (true, false) => "pk-mmap",
        (false, true) => "coset-spill",
        (true, true) => "pk-mmap+coset-spill",
    };
    (name, spill_pk)
}

fn assert_proof_verifies(
    srs: &ParamsKZG<Bls12>,
    pk: &midnight_proofs::plonk::ProvingKey<F, KZGCommitmentScheme<Bls12>>,
    instance: &[F],
    proof: &[u8],
) {
    let mut transcript = CircuitTranscript::<blake2b_simd::State>::init_from_bytes(proof);
    let trace = parse_trace(
        pk.get_vk(),
        &[&[C::identity()]],
        &[&[instance]],
        &mut transcript,
    )
    .expect("Failed to parse benchmark proof");
    let guard = verify_algebraic_constraints(
        pk.get_vk(),
        trace,
        &[&[C::identity()]],
        &[&[instance]],
        &mut transcript,
    )
    .expect("Benchmark proof failed algebraic verification");
    guard
        .verify(&srs.verifier_params())
        .expect("Benchmark proof failed opening verification");
}

fn bench_zswap_output(c: &mut Criterion) {
    let k = std::env::var("MIDNIGHT_BENCH_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(14);
    assert!((9..=25).contains(&k), "MIDNIGHT_BENCH_K must be in 9..=25");
    let srs = ParamsKZG::unsafe_setup(k, OsRng);

    let circuit = MidnightCircuit::from_relation(&ZSwapOutputCircuit, Some(k));
    let vk = keygen_vk_with_k::<_, KZGCommitmentScheme<Bls12>, _>(&srs, &circuit, k)
        .expect("Failed to generate VK");
    #[cfg_attr(not(feature = "disk-spill"), allow(unused_mut))]
    let mut pk = keygen_pk(vk, &circuit).expect("Failed to generate PK");
    let (instance, circuit) = sample_zswap_inputs(k);
    let (memory_profile, spill_pk) = benchmark_memory_profile(k);
    eprintln!("ZSwap benchmark memory profile: {memory_profile}");
    #[cfg(feature = "disk-spill")]
    if spill_pk {
        midnight_proofs::plonk::bench::prover::spill_proving_key(&mut pk)
            .expect("Failed to prepare mmap-backed proving key");
    }
    #[cfg(not(feature = "disk-spill"))]
    let _ = spill_pk;

    // This is intentionally a one-shot observation rather than a Criterion
    // distribution. It starts after the SRS and proving key are ready, so it
    // must not be confused with process/container cold start. Keeping it ahead
    // of the legacy phase benchmark prevents that benchmark from warming every
    // prover path before the first-proof number is captured.
    let mut first_proof = CircuitTranscript::<blake2b_simd::State>::init();
    let first_proof_started = Instant::now();
    create_proof(
        &srs,
        &pk,
        std::slice::from_ref(&circuit),
        1,
        &[&[&[], &instance]],
        &mut first_proof,
        OsRng,
    )
    .expect("First-proof observation failed to generate a proof");
    let first_proof_elapsed = first_proof_started.elapsed();
    assert_proof_verifies(&srs, &pk, &instance, &first_proof.finalize());
    eprintln!(
        "ZSwap first proof after key ready: k={k}, profile={memory_profile}, elapsed={first_proof_elapsed:?}"
    );

    let mut group = c.benchmark_group("ZSwap Prover");
    let mut transcript = CircuitTranscript::<blake2b_simd::State>::init();
    benchmark_create_proof(
        &srs,
        &pk,
        std::slice::from_ref(&circuit),
        1,
        &[&[&[], &instance]],
        &mut transcript,
        &mut OsRng,
        &mut group,
    )
    .expect("Failed to generate proof");

    group.finish();

    let mut group = c.benchmark_group("ZSwap Verifier");
    let transcript =
        CircuitTranscript::<blake2b_simd::State>::init_from_bytes(&transcript.finalize());
    group.bench_function("Parse trace", |b| {
        b.iter_batched(
            || transcript.clone(),
            |mut t| parse_trace(pk.get_vk(), &[&[C::identity()]], &[&[&instance]], &mut t).unwrap(),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("Verify algebraic constraints", |b| {
        b.iter_batched(
            || {
                let mut t = transcript.clone();
                (
                    parse_trace(pk.get_vk(), &[&[C::identity()]], &[&[&instance]], &mut t).unwrap(),
                    t,
                )
            },
            |(trace, mut t)| {
                verify_algebraic_constraints(
                    pk.get_vk(),
                    trace,
                    &[&[C::identity()]],
                    &[&[&instance]],
                    &mut t,
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("Finalize proof verification", |b| {
        b.iter_batched(
            || {
                let mut t = transcript.clone();
                let trace =
                    parse_trace(pk.get_vk(), &[&[C::identity()]], &[&[&instance]], &mut t).unwrap();
                let guard = verify_algebraic_constraints(
                    pk.get_vk(),
                    trace,
                    &[&[C::identity()]],
                    &[&[&instance]],
                    &mut t,
                )
                .unwrap();
                guard
            },
            |guard| guard.verify(&srs.verifier_params()).unwrap(),
            BatchSize::SmallInput,
        )
    });
    group.finish();

    // Unlike the legacy phase benchmark above, this measures the public,
    // end-to-end prover entry point. The line emitted above records which of
    // the production flags selected heap or mmap/spill storage for this process.
    // Run heap and spill in separate invocations: changing process-wide
    // environment variables between Criterion samples would be racy and would
    // produce an untrustworthy comparison.
    let mut group = c.benchmark_group(format!("ZSwap Prover End-to-end/k={k}"));
    group.sample_size(10);
    group.bench_function("steady-state-proof", |b| {
        b.iter_batched(
            CircuitTranscript::<blake2b_simd::State>::init,
            |mut transcript| {
                create_proof(
                    &srs,
                    &pk,
                    std::slice::from_ref(&circuit),
                    1,
                    &[&[&[], &instance]],
                    &mut transcript,
                    OsRng,
                )
                .expect("End-to-end benchmark failed to generate a proof");
                black_box(transcript.finalize())
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(200);
    targets = bench_zswap_output
);
criterion_main!(benches);
