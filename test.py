

file_list = [

    "090322.npy", "090332.npy", "090342.npy", "090352.npy", "090402.npy",

    "090412.npy", "090422.npy", "090432.npy", "090442.npy", "090452.npy",

    "090502.npy", "090512.npy", "091822.npy", "091832.npy", "091842.npy",

    "091852.npy", "091902.npy", "091912.npy", "091922.npy", "091932.npy",

    "091942.npy", "091952.npy", "092002.npy", "092012.npy", "091722.npy",

    "091732.npy", "091742.npy", "091752.npy", "091802.npy", "091812.npy",

    "091822.npy", "091832.npy", "091842.npy", "091852.npy", "091902.npy",

    "091912.npy"

]



# Initialize markdown content

markdown_content = ""

for i, filename in enumerate(file_list, start=1):

    index = str(i).zfill(2)

    markdown_content += f"""### Image '{filename}'

![](image/original_{index}.png)



### Hough Line

#### Intermediate steps of Hough Line

![](image/Intermidiate steps_{index}.png)



#### Final results Hough lines

![](image/Final results Hough lines_{index}.png)



### Linear Regression

#### Detected clusters

![](image/Detected clusters_{index}.png)



#### Detected lines Regression

![](image/Detected lines Regression_{index}.png)



---



"""



# Save the markdown content to a file

file_path = "./file_structure.md"

with open(file_path, "w") as f:

    f.write(markdown_content)
