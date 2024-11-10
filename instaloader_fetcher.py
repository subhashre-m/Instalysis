import instaloader

def fetch_instagram_captions(username, password=None):
    L = instaloader.Instaloader()
    
    # Perform login if password is provided
    if password:
        try:
            L.login(username, password)
        except instaloader.exceptions.BadCredentialsException:
            raise Exception("Invalid username or password.")
    
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        captions = [post.caption for post in profile.get_posts() if post.caption]
        return captions
    except instaloader.exceptions.LoginRequiredException:
        raise Exception("Login is required to fetch data for this account.")
    except instaloader.exceptions.ProfileNotExistsException:
        raise Exception(f"The profile '{username}' does not exist.")

def is_private_account(username, password=None):
    L = instaloader.Instaloader()
    
    # Perform login if password is provided
    if password:
        try:
            L.login(username, password)
        except instaloader.exceptions.BadCredentialsException:
            raise Exception("Invalid username or password.")
    
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        return profile.is_private
    except instaloader.exceptions.ProfileNotExistsException:
        raise Exception(f"The profile '{username}' does not exist.")
