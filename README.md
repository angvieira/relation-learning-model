# Theoretical Reference

## Neural Network

The neural network is a method used by artificial intelligence for processing data like the human cerebre. This method uses neurons connected in layers to analyze the data and learn with the error.
So, your architecture is a graph when have a input vector,  input potential, activation state and output. The neurons are interconnected via a set of directed, weighted connections. In figure xxx describe the neuron’s functionality, when the Wij are the weight, the x1(t),.., xn(t) are the inputs, Ai(t) is the activation state of the neuron, (Oi(t) is the output.

![image](https://user-images.githubusercontent.com/56411274/217053591-e2e95254-9628-40d1-8145-f4e3cd051953.png)

Figure 1 - The processing unit, or neuron

for activation the neuron are three ways: linear, nonlinear and semilinear (also known as sigmoid).

## Learning Strategy

![image](https://user-images.githubusercontent.com/56411274/217053536-3d0e89e9-163f-4585-b19f-41134e94a7ca.png)

Figure 2 - The processing unit, or neuron

# Problem
Michalski's train problem needs to classify the direction of the trains to east or west as shown in Figure xx. 

![image](https://user-images.githubusercontent.com/56411274/217053694-5575e3f8-3a9c-4a29-baf8-50dd4426f473.png)


This problem defined some rules:

 - for each train: 
	 - 	number of car  in 3 to 5 
	 - number of different load in 1 to 4 
 - for each car: 
	 - number of wheels in 2 ou 3 
	 - the length is short or long 
	 - The shape is a closed-top rectangle, open-top rectangle, double open rectangle, ellipse, engine, hexagon,jagged top, open trap, sloped top, or U-shaped. 
	 - the number of load in 0 to 3 
	 - the shape of the load is a (circle, hexagon, rectangle, or triangle) 
 - ten boolean describe pair of types of load are on adjacent cars of the train:
	 - there is a rectangle next to a rectangle (false or true),  
	 - a rectangle next to a triangle (false or true),  
	 - a rectangle next to a hexagon (false or true),  
	 - a rectangle next to a circle (false or true),  
	 - a triangle next to a triangle (false or true),  
	 - a triangle  next to a hexagon (false or true),  
	 - a triangle next to a circle (false or true),  
	 - a hexagon next to a hexagon (false or true),  
	 - a hexagon next to a circle (false or true), 
	 - a circle next to a circle (false or true).

# Solution

Link from Colab: https://colab.research.google.com/drive/1EnPTbl0xLE8vpR6O11NkUFqs_GjV5NlG#scrollTo=r_I1K8l_DjTO&uniqifier=1

## Database 

As a database, the following Train Data Set database was used, available at:

http://archive.ics.uci.edu/ml/datasets/Trains?ref=datanews.io

### Attribute Information:

The following format was used for the "transformed" dataset representation as found in trains.transformed.data (one instance per line):

**1.** Number_of_cars (integer in [3-5])
**2.** Number_of_different_loads (integer in [1-4])
**3-22:** 5 attributes for each of cars 2 through 5: (20 attributes total)
- num_wheels (integer in [2-3])
- length (short or long)
- shape (closedrect, dblopnrect, ellipse, engine, hexagon, jaggedtop, openrect, opentrap, slopetop, ushaped)
- num_loads (integer in [0-3])
- load_shape (circlelod, hexagonlod, rectanglod, trianglod)
**23-32:** 10 Boolean attributes describing whether 2 types of loads are on adjacent cars of the train
- Rectangle_next_to_rectangle (0 if false, 1 if true)
- Rectangle_next_to_triangle (0 if false, 1 if true)
- Rectangle_next_to_hexagon (0 if false, 1 if true)
- Rectangle_next_to_circle (0 if false, 1 if true)
- Triangle_next_to_triangle (0 if false, 1 if true)
- Triangle_next_to_hexagon (0 if false, 1 if true)
- Triangle_next_to_circle (0 if false, 1 if true)
- Hexagon_next_to_hexagon (0 if false, 1 if true)
- Hexagon_next_to_circle (0 if false, 1 if true)
- Circle_next_to_circle (0 if false, 1 if true)
**33.** Class attribute (east or west)
The number of cars varies between 3 and 5. Therefore, attributes referring to properties of cars that do not exist (such as the 5 attriubutes for the "5th" car when the train has fewer than 5 cars) are assigned a value of "-"

### Examples:
5 4 2 long openrect 3 rectanglod 2 short slopetop 1 trianglod 3 long openrect 1 hexagonlod 2 short openrect 1 circlelod 0 1 0 0 0 1 0 0 1 0 east
3 1 2 short ushaped 1 rectanglod 2 long openrect 2 rectanglod - - - - - - - - - - 1 0 0 0 0 0 0 0 0 0 west

### Implementation

#### Neural Network

For trainning, test and implementation for the neural networks the code uses the TensorFlow libary.

```
import tensorflow as tf
import tensorflow.keras as keras

#creating a neural network
inputs = keras.Input(shape=(tam,), name="digits")
x = tf.keras.layers.Dense(20, activation=bipolar_semilinear, name="dense_1")(inputs)
#x = tf.keras.layers.Dense(9, activation="relu", name="dense_2")(x)
outputs = tf.keras.layers.Dense(1, activation=bipolar_semilinear, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

#### Processing the data

With the trains-transformed.data file the code divides the data according to the inputs of the neural networks.

![image](https://user-images.githubusercontent.com/56411274/217673332-e0b54365-83de-49ac-9c9a-b43c26ac2731.png)


## The neural networks

The code have 11 neural networks, defined by:
num_cars(t, nc), in which t ∊ [1..10] and nc ∊ [3..5].
num_loads(t, nl) in which t ∊ [1..10] an nl ∊ [1..4].
num_wheels(t, c, w) in which t ∊ [1..10] and c ∊ [1..4] e w ∊ [2..3].
length(t, c, l) in which t ∊ [1..10] and c ∊ [1..4] and l ∊ [-1..1] (-1 denotes short and 1 long)
shape(t, c, s) in which t ∊ [1..10] and c ∊ [1..4] and s ∊ [1..10] (one number for each shape).
num_cars_loads(t, c, ncl) in which t ∊ [1..10] and c ∊ [1..4] e ncl ∊ [0..3].
load_shape(t, c, ls) in which t ∊ [1..10] e c ∊ [1..4] and ls ∊ [1..4].
next_crc(t, c, x) in which t ∊ [1..10] and c ∊ [1..4] and x ∊ [-1..1], in which car c of train t has an adjacent car with loads in circle form.
next_hex(t, c, x) in which t ∊ [1..10] and c ∊ [1..4] and x ∊ [-1..1], in which car c of train t has an adjacent car with loads in a hexagon shape.
next_rec(t, c, x) in which t ∊ [1..10] e c ∊ [1..4] e x ∊ [-1..1], in which car c of train t has an adjacent car with rectangle loads.
next_tri(t, c, x) in which t ∊ [1..10] e c ∊ [1..4] e x ∊ [-1..1], in which car c of train t has an adjacent car with triangle loads

# Results

# References
