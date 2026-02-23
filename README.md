# Computer-Vision-Photobooth-Mac
Welcome to our Virtual Photobooth! Because this project utilizes heavy machine learning libraries (mediapipe, rembg, opencv-python e.t.c.) that are highly sensitive to specific Python versions and environment conflicts, this project uses uv for dependency management.

[uv](https://docs.astral.sh/uv/) is an ultra-fast Python package manager that will automatically download the correct, isolated version of Python (3.10) required for Google's MediaPipe, guaranteeing a crash-free setup regardless of your computer's default Python version.

## Set up
### Step 1: Install uv

If you do not already have uv installed on your system, open your terminal and run the following command:

- Mac/Linux: ```curl -LsSf https://astral.sh/uv/install.sh | sh```

- Windows: ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```


### Step 2: Create the Virtual Environment

Navigate to this project folder in your terminal. Run the following command to create a virtual environment explicitly using Python 3.10 (This specific version is required for MediaPipe 0.10.14 compatibility):

```
uv venv --python 3.10 .venv
```

### Step 3: Activate the Environment

Before installing packages, you must activate the environment:

- Mac/Linux: ```source .venv/bin/activate```

- Windows: ```.venv\Scripts\activate```
    (You should see (.venv) appear at the start of your terminal prompt).

### Step 4: Install Dependencies
With the environment activated, install the required libraries. ```uv``` will resolve these in seconds:
```
uv pip install -r requirements.txt
```

## Controls & Features

- 1-5: Toggle through different AR Props (Hats, Glasses, Masks)
- 6-9: Toggle UI Elements and Effects (Frames, Confetti, Halo Streaks)
- 0: Cycle through Virtual Backgrounds (Requires Virtual BG to be active)
- SPACE: Capture Photo (Triggers high-quality rembg background removal)
- Q: Quit Application

## Write up
"After our 4 years in SST, we have finally reached the SST Graduation Tea. With all that we have been through, we have made many lasting friends and memories. For this Graduation Tea, we will have a virtual photobooth with many props and filters that pay homage to our unique SST identity and culture while reflecting on the many events in the 4 years of SST. These props can be worn on the head, face, and body, with a large selection for each category. Our photobooth will also have many effects, filters and props to complement and accommodate users' personal style. We created this photobooth to help SST Students take a memorable photo that they can carry throughout their journey of life.

1. The graduation cap signifies the students' end at SST and their completion of secondary school, a 4-year journey of suffering, enjoyment and achievement.
2. The propeller hat symbolises how the students of SST first started small and spun slowly. Still, as the students gained more knowledge and skills from project work and assessments, the propeller became fast enough to take them out of SST and into the big outside world. 
3. The SST 26 glasses both show the graduating year of the SST students and the students' school. Because each year that passes by never comes back, each batch of SST students has its unique SST glass of the year that other batches will not have. 
4.The sunglasses are meant to show how cool and tough SST students are after 4 years of journeying with SST, their teachers, and their peers, and how that has made them who they are. 
5.The beard doesn't grow overnight; it grows step by step, follicle by follicle, showing the incremental growth of SST students as they endured the hardships of late-night PT and project deadlines. It shows how the SST becomes a rugged, experienced "Senior" at the school, a testament to hard work and dedication.
6. The confetti celebrates the milestone of SST students and creates a happy atmosphere as they mark the last school year at SST and take a memorable photo in the best setting at the moment, with what they have. The different backgrounds also allow the student to take a precious photo at their chosen spot, which is essential to them during their journey in SST, or at a place where all their friends are taking pictures, giving them a strong sense of belonging.
 
For the other features, we also have a three-second timer with a built-in flash, which will improve the photo-taking experience by allowing students to position themselves better after pressing the button, as well as add an extra layer of immersion. We also have an advanced tracking system that allows us to track the user's head, enabling the props to tilt and follow the user's body more naturally. Additionally, our model utilised a hybrid approach: the people displayed are rendered using a lighter model, and the actual photo is saved using a heavier, more accurate model, allowing the photobooth to feel smooth while delivering a professional experience with a studio-like picture.

Lastly, the background we have thoughtfully selected consists of familiar places SST students have been to and best represents their time at SST. The photo choices allow students to choose their most special place in SST to take a picture. Additionally, there is a plain background for students who just want a simple photo with their classmates and friends. 
