# eeg_analysis

graph LR
    A[Raw EEG Data .edf/.mat] --> B{Load Data}
    B -->|Extract Signal & fs| C{Preprocessing Required?}
    
    %% Preprocessing Flow
    C -->|Yes| D[Bandpass Filter]
    D --> E[Artifact Removal ASR]
    E --> F[Cleaned EEG Signal]
    
    C -->|No| F
    
    %% Feature Extraction Flow
    F --> G[Extract Features Channel-by-Channel]
    
    subgraph Feature Extraction Module
        G --> H[FOOOF Theta Peak Extraction]
        H --> I[Append Extracted Metadata]
    end
    
    %% Output Flow
    I --> J[Aggregate Channel Data]
    J --> K[Export as CSV File]
