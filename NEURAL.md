# Neural network experiment

Across nine held-out Graz participants, the expanded neural network classified 1,574 of 2,346 trials correctly (67.1%). CSP trained on the same expanded data classified 1,530 correctly (65.2%). The network scored higher for six of the nine participants. The original-data neural network scored 66.2%, so the 1,023 external trials contributed 22 additional correct predictions on net. The previously inspected A02–A09 subset scored 65.0% with the expanded network versus the original 64.2% CSP result. These scores describe this exploratory benchmark. The final model trained on all 33 participants has no independent test in this experiment.

Run these commands from the project folder using the existing project environment.

```powershell
.\.venv\Scripts\python.exe fetch_neural_data.py
.\.venv\Scripts\python.exe train_neural.py
```

The experiment compares a feedforward neural network and the existing four-component CSP and shrinkage LDA classifier. Each of nine rounds excludes one Graz participant and both of their sessions. The original-data arms train on the other eight people. The expanded arms also train on PhysioNet participants S001 through S024. Participant IDs, frequency bands, network size and the 30-epoch training budget were fixed before scoring the test people. All input scaling learned from data is fitted inside the training fold.

The download script retrieves runs 4, 8 and 12 from each external participant and checks each EDF against the SHA-256 registry bundled with MNE. Those particular runs contain imagined left and right fist movements. T1 maps to left and T2 to right. The external signals use the 22 electrode positions shared with Graz, in the Graz order, and are resampled from 160 Hz to 250 Hz. We require a complete five-second window and an imagery annotation that covers at least 3.5 seconds. The datasets used different acquisition protocols. The PhysioNet data do not provide the same expert artifact flags as the Graz files.

For each model the scoring window runs from 0.5 to 3.5 seconds after the cue, with padding for filtering. CSP uses the project's original 8–30 Hz pipeline. The neural network uses 108 relative log-power and left/right asymmetry features from four bands between 8 and 30 Hz. It has one hidden layer of 32 ReLU units, trained using Adam. Its feature extractor applies common-average referencing and local electrode differences around C3, Cz and C4. This is a network on EEG features. All reported neural trials are scored using the model that excluded that person. The final development model additionally trains on all nine Graz participants and all 24 external participants.

The original CSP predictions come from the previous verified multi-participant experiment and are joined by recording and original trial ID. The expanded CSP model is fitted on the same additional trials as the expanded neural network. The A02–A09 subset supports comparison with the previously reported 64.2% CSP result. All nine participants form the primary comparison.

The experiment writes trial predictions, correct counts by person, external trial counts by recording, nine held-out models, a final development model and a JSON summary in `outputs/neural/`. The saved bundle's `pipeline.predict` accepts five-second raw EEG epochs in volts shaped (trials, 22, 1250) in the saved channel order, with the cue at sample 125. The all-person model requires an independent cohort for an accuracy estimate. The benchmark recordings were previously examined.

External data are the **EEG Motor Movement/Imagery Dataset v1.0.0**, Gerwin Schalk (2009), published on PhysioNet under ODC Attribution v1.0. BCI2000 is described by Schalk et al. (2004), IEEE Transactions on Biomedical Engineering. [Dataset, full citations and license](https://physionet.org/content/eegmmidb/1.0.0/) · [Dataset DOI](https://doi.org/10.13026/C28G6P) · [Graz electrode ordering](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/bnci_2014.py)
