use serde::Serialize;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("Something went wrong: {0}")]
    Failed(String),
    #[error("Vector not found")]
    NotFound,
    #[error("Table does not exist")]
    TableNotExist,
    #[error("Not implemented")]
    NotImplemented,
    #[error(transparent)]
    SerializationFailed(#[from] std::boxed::Box<bincode::ErrorKind>),
    #[error(transparent)]
    SqlFailure(#[from] rusqlite::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
