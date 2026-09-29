use rustc_hash::FxHashMap;

fn main() {
    let mut map = FxHashMap::default();
    map.insert(1, 2);
    println!("{:?}", map);
}
