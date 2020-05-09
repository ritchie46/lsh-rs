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
    #[cfg(feature = "sqlite")]
    SqlFailure(#[from] rusqlite::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
