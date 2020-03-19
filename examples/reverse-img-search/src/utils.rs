use std::fs;

pub fn file_iter(vec_folder: &str) -> Box<dyn Iterator<Item = Vec<f32>>> {
    let a = fs::read_dir(vec_folder).unwrap().map(|entry| {
        let entry = entry.unwrap();
        let f = fs::File::open(entry.path()).unwrap();
        let mut v: Vec<f32> = serde_cbor::from_reader(f).unwrap();
        v
    });
    Box::new(a)
}
