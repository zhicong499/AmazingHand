Project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)


Mechanical design is licensed under a:
[Creative Commons Attribution 4.0 International License][cc-by].
[![CC BY 4.0][cc-by-image]][cc-by]
[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY-lightgrey.svg



# Amazing Hand + 


Based on Amazing Hand initial project, here is an update with new motors which will offer :
- Full backdrivability
- Better accuracy
- More Speed
- Same overall volume

Thanks to the use of Feetech STS3032 motors, Amazing Hand + pushes the boundaries of the original version.

In return, its price will also increase… but will still remain reasonable compared to most other robotic hands available !



## Table of contents

- [Modifications](#modifications)
    - [BOM (Bill Of Materials)](#bom-bill-of-materials)
    - [CAD Files and Onshape document](#cad-files-and-onshape-document)
    - [Assembly Step](#assembly-steps)
    - [Run_basic_Demo](#run-basic-demo)    
- [AmazingHand+ advanced Demo](#amazinghand+-advanced-demo)   


# Build Resources
## BOM (Bill Of Materials)

All standard components are the same, except motors.

List of all needed components is available here:  [AmazingHand+ BOM](https://docs.google.com/spreadsheets/d/180ta_jQhF-YGHinabTnSPGtvGcQORVJ9lFrNiKcSqUE/edit?gid=1269903342#gid=1269903342)  
![BOM](assets/BOM+.jpg)


## CAD Files and Onshape document

All fingers parts are the same for both version : Proximal / Distal / Gimbal / Link

=> Please refer to original version for all these parts [here](https://github.com/pollen-robotics/AmazingHand/tree/main/cad) 

Specific parts for Amazing Hand + :
- Finger frame part 1 & 2 become Finger frame (only 1 part)
- Custom servo horn is needed, to be 3D printed
- Hand Plate / Wrist interface / Palm shell & Top shell are slightly differen (due to custom servo horn and increased servo thickness)

=> Specific STL and Steps files for Amazing Hand+ can be found [here](https://github.com/pollen-robotics/AmazingHand/tree/AmazingHand+/cad) 

Everyone can access the Onshape document of Amazing Hand+ :   [Link Onshape](https://cad.onshape.com/documents/430ff184cf3dd9557aaff2be/w/b018c00d2300dcc36080a900/e/a5789448a382c9e7e5c5750a)  


## Assembly Steps

First change is on "Step 1 : Preparing components" , slide 6. Some screws (coming with ball joints) length need to be adjusted, and custom servo horn need to be prepared too.

Main changes on assembly is on "Step 2 : Finger assembly" , slides 12, 17 & 18. Now there is only 1 finger frame part, and more screws to hold motors

"Step 3 : finger calibration" remains the same principle, but with slightly different max angles. Details explain on slides 22 & 23.

Assembly guide for the Amazing Hand + in combination with standard components in the BOM is here:  
![Assembly Guide](/docs/AmazingHand+_Assembly.pdf)  
![Assembly_example](/assets/Assembly+.jpg)   

In addition, a very annoying ultrasound could be heard by some of you, because of the pwm frequency set to 16KHz by defaut. This frequency could be changed to 24KHz by modifying the register 18 "seting byte" to 44, and save (using feetech software). This may need a servo reboot to be applied then.
![PWM frequency](assets/PWM-frequency.jpg)


## Run basic Demo

A basic demo is available with both Python.

- Python script: "AmazingHand_Demo.py" [here](https://github.com/pollen-robotics/AmazingHand/tree/AmazingHand+/PythonExample)



# AmazingHand+ advanced Demo
To be released

