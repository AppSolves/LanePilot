<div align="center">

<img src="./assets/github/readme-anim.gif" alt="readmeanim" width="400" height="400"/>

-----

Check out these examples of videos generated using `QuickClipAI` (by [CurioBurstz](https://www.youtube.com/@curioburstz?sub_confirmation=1)):

<a href="http://www.youtube.com/watch?v=p9FsprPymo4" title="Why You Feel Pain: The Fascinating Science Behind Your Body's Alarm System">
  <img src="http://img.youtube.com/vi/p9FsprPymo4/0.jpg" alt="CurioBurstz No. 1" width="300" height="225" hspace="25">
</a>
<a href="http://www.youtube.com/watch?v=1ObvKfjU0Vg" title="The Mystery of Déjà Vu: Unraveling the Mind's Glitch">
  <img src="http://img.youtube.com/vi/1ObvKfjU0Vg/0.jpg" alt="CurioBurstz No. 2" width="300" height="225">
</a>

-----

# `QuickClipAI`

![GitHub stars](https://img.shields.io/github/stars/AppSolves/QuickClipAI)
![GitHub forks](https://img.shields.io/github/forks/AppSolves/QuickClipAI)
![GitHub issues](https://img.shields.io/github/issues/AppSolves/QuickClipAI)
![GitHub license](https://img.shields.io/github/license/AppSolves/QuickClipAI)

<h4><code>QuickClipAI</code> is a tool that uses AI to automatically create engaging YouTube shorts and videos from text inputs.

Sit back and relax while <strong>QuickClipAI</strong> boosts your productivity!</h4>

[Introduction](#introduction-) • [Features](#features-) • [Installation](#installation-%EF%B8%8F) • [Usage](#usage-) • [Credits](#credits-) • [License](#license-)

</div>

<br />

> 👋 This is a very early release and _lots_ more documentation and functionality is currently being added.

# QuickClipAI 🤖

## Introduction 📖
Welcome to the **QuickClipAI** Project! This project aims to provide an efficient and powerful AI-driven tool for generating engaging YouTube shorts and videos automatically. Whether you're a content creator, marketer, or just someone looking to boost your online presence, QuickClipAI offers a seamless and creative video generation solution.

## Features 🚀
- **AI-Driven Content Generation**: Leverages advanced AI algorithms to automatically create engaging YouTube shorts and videos from text inputs.
- **Customizable Video Settings**: Allows users to adjust various parameters such as length, style, and theme to suit their content needs.
- **High-Quality Output**: Ensures videos are generated with professional quality, ready for upload to YouTube or other platforms.

## Installation 🛠️

### Binaries & Packages 📦
1. **Clone the Repository**: Clone the repository to your local machine using:

```bash
git clone https://github.com/AppSolves/QuickClipAI.git
```

2. **Install Dependencies:** Navigate to the root directory and install the required libraries using:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Install `Fooocus`**: Download the latest release of [Fooocus](https://github.com/lllyasviel/Fooocus#download) for your operating system and specify it's path within `main.py` before any other function calls: 

> ⚠️ **IMPORTANT**: This project was developed and tested using `Fooocus` version `2.5.5`. If you encounter any issues with other versions, please install the specified version.

```python
# Running this will save the path to the `config/config.json` file.
# Once the path is saved, you can and should run the script without those lines.

settings_manager = SettingsManager()
settings_manager.set(
    "fooocus_path",
    {
        "interpreter": r"path\to\fooocus\embeded\python\interpreter", # Can be 'python3' on Unix systems, while Windows systems require the full path to the embedded interpreter.
        "script": r"path\to\fooocus\launch.py",
    },
    encrypt=False, # You can set this to True if you want to encrypt sensitive data.
)
```

4. **Install `ffmpeg`**: Download and install `ffmpeg` from the [official website](https://ffmpeg.org/download.html). Make sure to add the binary to your system's `PATH` variable.

5. **Install `ImageMagick`**: Download and install `ImageMagick` from the [official website](https://imagemagick.org/script/download.php) as well. Make sure to tick both boxes that
- add the binary to your system's `PATH` variable.
- install the legacy utilities (e.g. `convert.exe`).

6. **Install `Chrome`**: `Chrome` and `ChromeDriver` are necessary if you would like to use background music from [Bensound](https://www.bensound.com) (refer to the `BensoundBackgroundMusic` class) or make automatic uploads to [Instagram](https://instagram.com) and [TikTok](https://tiktok.com). Make sure to download the latest version of [Chrome](https://www.google.com/chrome/) and the [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/#stable) respectively. Make sure to add the `ChromeDriver` binary to your system's `PATH` variable.

### Providers and API setup 🌐
1. You need to obtain an API key from [ElevenLabs](https://elevenlabs.io) (they have a free plan) to be able to generate automatic voiceovers. You can set this key in the same `.env` file as above by adding the `ELEVENLABS_API_KEY` key or by setting it using the `SettingsManager` class:

```python
# Again, you may remove these lines after executing the script once.

settings_manager = SettingsManager()
settings_manager.set(
    "elevenlabs_api_key",
    "<your_api_key>",
    encrypt=True, # Encryption is recommended for sensitive data.
)
```

You can also set the API key using the command line interface (`CLI`) as follows (better for shorter data):

```bash
python main.py settings set "elevenlabs_api_key" "<your_api_key>" -e
```

2. In order to generate text, the `G4FAPI` uses `g4f.Provider.OpenaiChat` as the default provider. You can change this by passing a custom provider to the `G4FAPI` class. The setup of `.har` files is necessary for authentication. You can set these files in the `config/har_and_cookies` directory and the `G4FAPI` class will automatically use them:

> ⚠️ **IMPORTANT**: In order to function correctly, some providers, e.g. "you.com", require cookies to be set. You can set these cookies by generating a `.har` file for the provider's website using your browser's developer tools and placing it in the `config/har_and_cookies` directory. If `ENCRYPTION_KEY` is set, the `.har` file will be encrypted when the program is run for the first time and only decrypted during runtime.

Source: Official [gpt4free](https://github.com/xtekky/gpt4free#har-file-for-openaichat-provider) repository. Example for `chatgpt.com` and `g4f.Provider.OpenaiChat` respectively:

> To utilize the OpenaiChat provider, a .har file is required from https://chatgpt.com/. Follow the steps below to create a valid .har file:
>   1. Navigate to https://chatgpt.com/ using your preferred web browser and log in with your credentials.
>   2. Access the Developer Tools in your browser. This can typically be done by right-clicking the page and selecting "Inspect," or by pressing F12 or Ctrl+Shift+I (Cmd+Option+I on a Mac).
>   3. With the Developer Tools open, switch to the "Network" tab.
> 4. Reload the website to capture the loading process within the Network tab.
> 5. Initiate an action in the chat which can be captured in the .har file.
> 6. Right-click any of the network activities listed and select "Save all as HAR with content" to export the .har file.

In order to make use of the automatic upload features, you need to add your YouTube credentials to the `config/config.json` file using the `SettingsManager` class. The `UploadAPI` class will automatically use these credentials to upload the generated video to the respective platform:

```python
# You SHOULD remove these lines after executing the script once.

settings_manager = SettingsManager()
settings_manager.set(
    "publisher",
    {
        "youtube": {
            "installed": {
                "client_id": "<some-id>.apps.googleusercontent.com",
                "project_id": "<some-name>",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": "<some-secret>",
                "redirect_uris": ["http://localhost"],
            }
        },
    },
    encrypt=True, # You should definitely encrypt sensitive data.
)
```

> ⚠️ **IMPORTANT**: For Instagram and TikTok, you only have to be logged in to the respective platform in your chrome browser.
>
> It may be necessary to occasionally resolve captchas on [TikTok](https://www.tiktok.com/tiktokstudio) when using the TikTok provider until a more permanent solution is found (e.g. automatic captcha solving).

Follow these guides to setup the publisher credentials:
- Set up a Google Cloud Project for [YouTube](https://developers.google.com/youtube/v3/guides/uploading_a_video)

## Customization 🎨

QuickClipAI offers a range of customization options to tailor the generated videos to your specific needs. By now, you can adjust all prompts in the `prompts` folder to tweak the generated content, add custom background music in the `assets/project/music` folder and add custom fonts in the `assets/project/fonts` folder. More customization options will be added in future releases.

> ⚠️ **IMPORTANT**: By now, only `.ttf` fonts are supported. Make sure to add your custom fonts in the `assets/project/fonts` folder and run the following command in admin mode to register them:
>
>```bash
>captametropolis register_font "assets\project\fonts\your_font.ttf" -qr # If not run in admin mode, the command will ask for admin permissions.
>```
>
> This will register the font in the system and make it available for use in the video generation process. Make sure to not change the font name or location after registering it.

> :information_source: **NOTE**: You can easily add your own background music by placing the file in the `assets/project/music` folder and using the `background_music` parameter in the `MoviepyAPI().generate_video()` function:
> 
> ```python
> moviepy_api = MoviepyAPI()
> moviepy_api.generate_video(
>     ...,
>     background_music=BackgroundMusic(
>         r"assets\project\music\your_music.mp3",
>         start_sec=0,
>         volume_factor=0.2,
>         credits="Credits to song artist", # Optional
>     ),
> )
> ```
>
> You can even use sound driectly from [Bensound](https://www.bensound.com) using the `BensoundBackgroundMusic` class, it will automatically generate the credits for you:
>
> ```python
> moviepy_api.generate_video(
>     ...,
>     background_music=BensoundBackgroundMusic(
>         "the lounge", # Search query for the song
>         start_sec=0,
>         volume_factor=0.2,
>     ),
> )
> ```

### Encryption 🔐

If you would like to use encryption features for the storage of configuration files, you can enable them by creating a `.env` file in the `config` directory and adding the `ENCRYPTION_KEY` key along with your password. The `SettingsManager` class currently supports encryption for `str`, `bytes`, `dict`, and `list` types. You can set the key in the `.env` file as follows:

```env
ENCRYPTION_KEY=your_encryption_key
```

### Upload to YouTube and Instagram 📤

You can easily upload your generated videos to YouTube and Instagram automatically using the `UploadAPI` class. You can set the `youtube` parameter to `True` to upload the video to YouTube, and the `instagram` and `tiktok` parameters to `True` to upload the video to Instagram and TikTok respectively.

You can then use the `UploadAPI` class to upload the video:

```python
upload_api = UploadAPI()
upload_api.upload_video(
    session_id="...", # The session ID of the video to upload (will be displayed in the console after generating the video). Leave `None` to use the last session ID.
    youtube=True,
    instagram=True,
    tiktok=True,
)
```

## Usage 📝
1. **Activate Virtual Environment**: Navigate to the root directory and activate the virtual environment using:

```bash
source venv/bin/activate
```

2. **Configure VPN**: If you are using a VPN, make sure to switch the location/server regularly to avoid being blocked by the `g4f.Provider`. If you encounter any issues with the provider, you can set the `provider` parameter in the `G4FAPI` class to `None`, which will force the API to choose the best provider available automatically:

```python
g4f_api = G4FAPI(provider=None)
```

3. **Run the Script**: Run the main script using:

```bash
python main.py
```

## Credits 🙏
This project was developed and is maintained by [AppSolves](https://github.com/AppSolves).

#### Links

- [E-Mail](mailto:contact@appsolves.dev)
- [Website](https://appsolves.dev)
- [GitHub](https://github.com/AppSolves)
- [This Repository](https://github.com/AppSolves/QuickClipAI)

Also, check out `CurioBurstz` on [YouTube](https://www.youtube.com/@curioburstz?sub_confirmation=1), [Instagram](https://www.instagram.com/curioburstz/), and [TikTok](https://www.tiktok.com/@curioburstz) for more content created using `QuickClipAI`.

## License 📜
This project is licensed under a custom license with **All Rights Reserved**.  
No use, distribution, or modification is allowed without explicit permission from the author.

For more information, please see the [LICENSE](LICENSE.txt) file.

QuickClipAI © 2024 by Kaan Gönüldinc

## Conclusion 🎉

Thank you for checking out QuickClipAI! We hope you find this tool useful for your content creation needs. If you have any questions, feedback, or suggestions, please feel free to reach out to us.