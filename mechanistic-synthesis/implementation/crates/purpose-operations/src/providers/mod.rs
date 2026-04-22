//! Built-in providers.

pub mod summary;
pub mod uniprot;

pub use summary::ProteinSummaryProvider;
pub use uniprot::UniprotProvider;
