
=======
    #print("Len training_example_eq8s dimension 1: " + str(len(training_example_eq8s)))
    print("Len training_example_eq8s dimension 2: " + str(len(training_example_eq8s[0])))
    print("Len training_example_eq8s dimension 3: " + str(len(training_example_eq8s[0][0])))
    #print(training_example_eq8s[0])
    #exit()

    eq16 = np.zeros((len(training_example_eq8s[0]), len(training_example_eq8s[0][0])))
    for training_example_index in range(len(training_data)):
        eq16 = np.add(training_example_eq8s[training_example_index], eq16)

    w2T = np.multiply(lambdaval, w2.T)

    eq16 = np.add(eq16, w2T)

    eq16 = np.divide(eq16, len(training_data))
>>>>>>> Stashed changes

    print("eq16 shape: " + str(eq16.shape))
    print("eq16: " + str(eq16))
