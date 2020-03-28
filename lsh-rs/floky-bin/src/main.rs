use lsh_rs::LshSql;
use std::env;
use std::fs::File;
use std::io;
use std::io::{BufRead, Write};
use std::path::Path;

fn usage() {
    println!(
        "
floky-bin <n-projections> <n-hash-tables> file.csv
    "
    )
}

fn read_csv<P>(path: P) -> Vec<Vec<f32>>
where
    P: AsRef<Path>,
{
    let mut vs = vec![];
    if let Ok(lines) = read_lines(path) {
        for line in lines {
            if let Ok(line) = line {
                let mut split = line.split(',');

                let mut v = vec![];
                for s in split {
                    let val: f32 = s.parse().expect("could not parse values");
                    v.push(val)
                }
                vs.push(v)
            }
        }
    };
    vs
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn run_lsh(n_projections: usize, n_hash_tables: usize, vs: &Vec<Vec<f32>>) {
    let dim = vs[0].len();
    let mut lsh = LshSql::new(n_projections, n_hash_tables, dim)
        .only_index()
        .l2(4.)
        .expect("could not make lsh");

    let total = vs.len();
    let mut c = 0;
    for chunk in vs.chunks(100) {
        print!("{}/{}\r", c, total);
        std::io::stdout().flush();
        lsh.store_vecs(chunk);
        c += 100;
        lsh.commit();
        lsh.init_transaction();
    }
    lsh.commit();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        4 => {
            let default = String::from("18");
            let n_projections: usize = args
                .get(1)
                .unwrap_or(&default)
                .parse()
                .expect("n-projections not properly defined");
            let default = String::from("20");
            let n_hash_tables: usize = args
                .get(2)
                .unwrap_or(&default)
                .parse()
                .expect("n-hash-tables not properly defined");

            let csv = args.get(3).expect("file not given");
            let vs = read_csv(csv);
            run_lsh(n_projections, n_hash_tables, &vs);
        }
        _ => {
            usage();
        }
    }
}
