# WITHOUT BALANCE DATASET
df_naive = df[['census_carrier_name', 'plan_admin_name', 'carrier_name', 'name']].copy()

# Combine the text columns into one, as we will treat them as one input to the network
df_naive['combined_text'] = df_naive.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Assuming 'is_match' is the binary target column you want to predict
y = df['is_match'].values  # Or any other column that is the target variable

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df_naive['combined_text'])
sequences = tokenizer.texts_to_sequences(df_naive['combined_text'])

# Pad sequences to ensure uniform input size
max_sequence_length = max(len(x) for x in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_sequence_length))
model.add(Flatten())  # Flatten the output of the embedding layer to feed into the dense layer

# Use one neuron with 'sigmoid' activation function for binary classification
model.add(Dense(10, activation='relu'))  # Hidden layer with 10 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with one neuron for binary classification

# Compile the model with binary_crossentropy for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# WITH BALANCE DATASET
df_naive = df[['census_carrier_name', 'plan_admin_name', 'carrier_name', 'name']].copy()

# Combine the text columns into one, as we will treat them as one input to the network
df_naive['combined_text'] = df_naive.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Assuming 'is_match' is the binary target column you want to predict
y = df['is_match'].values  # Or any other column that is the target variable

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df_naive['combined_text'])
sequences = tokenizer.texts_to_sequences(df_naive['combined_text'])

# Pad sequences to ensure uniform input size
max_sequence_length = max(len(x) for x in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_sequence_length))
model.add(Flatten())  # Flatten the output of the embedding layer to feed into the dense layer

# Use one neuron with 'sigmoid' activation function for binary classification
model.add(Dense(10, activation='relu'))  # Hidden layer with 10 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with one neuron for binary classification

# Compile the model with binary_crossentropy for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate class weights (imbalance dataset)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, class_weight=class_weight_dict)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")