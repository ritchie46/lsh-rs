pub use crate::{
    error::{Error, Result},
    hash::{SignRandomProjections, VecHash, L2, MIPS},
    lsh::lsh::LSH,
    table::{general::HashTables, mem::MemoryTable, sqlite::SqlTable, sqlite_mem::SqlTableMem},
};

pub type LshSql<H, N = f32, K = i8> = LSH<H, N, SqlTable<N, K>, K>;
pub type LshSqlMem<H, N = f32, K = i8> = LSH<H, N, SqlTableMem<N, K>, K>;
pub type LshMem<H, N = f32, K = i8> = LSH<H, N, MemoryTable<N, K>, K>;

macro_rules! concrete_lsh_structs {
    ($mod_name:ident, $K:ty) => {
        pub mod $mod_name {
            use super::*;
            pub type LshSql<H, N = f32> = LSH<H, N, SqlTable<N, $K>, $K>;
            pub type LshSqlMem<H, N = f32> = LSH<H, N, SqlTableMem<N, $K>, $K>;
            pub type LshMem<H, N = f32> = LSH<H, N, MemoryTable<N, $K>, $K>;
        }
    };
}
concrete_lsh_structs!(hi8, i8);
concrete_lsh_structs!(hi16, i16);
concrete_lsh_structs!(hi32, i32);
concrete_lsh_structs!(hi64, i64);
