import tarfile
import urllib.request

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
file_name = "gdb9.tar.gz"

# Download the file
urllib.request.urlretrieve(url, file_name)

# Extract the tar.gz file
with tarfile.open(file_name, "r:gz") as file:
    file.extractall()

# Now, you can access "gdb9.sdf" inside the extracted directory
