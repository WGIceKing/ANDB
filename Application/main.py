import arcade
import arcade.gui
import random as r
import os
from model_app import *

# Constants for window options 
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
USERS_NUM = 5

class App(arcade.Window):
    def __init__(self):
        # Window intialization 
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, 'Testing')
        # Initial values assignment 
        # screen_num 0 -> films selection, 1 -> titles displaying 
        self.buttons_list = None
        self.users_list = []
        self.previous = -1
        self.clicked = -1
        self.screen_num = 0
        self.results = []

    def on_draw(self):
        # Drawing elements in the window
        arcade.start_render()
        if (self.screen_num == 0):
            # Text on the top of the screen
            arcade.draw_text("Select list of films that you find interesting",
                            0,SCREEN_HEIGHT-40,
                            width=SCREEN_WIDTH,
                            font_size=24,
                            align="center")
            # Drawing buttons 
            self.buttons_list.draw()
            for i in range(USERS_NUM):
                # Drawing lists of top 5 rated films assigned to specific user
                temp_df = ratings_df.loc[(ratings_df['userId']==str(self.users_list[i]))].sort_values(by=['rating'],ascending=False).head()
                titles = temp_df['original_title'].values.tolist()
                for j in range(len(titles)):
                    # Drawing single title
                    arcade.draw_text(f"{j+1}. {titles[j]}",
                                175,SCREEN_HEIGHT/(USERS_NUM+2) * i + 172 - (j*14),
                                arcade.color.BLACK,10)
            # Submit and refresh buttons
            submit = arcade.Sprite("submit.png",0.5)
            submit.center_x,submit.center_y = 500, 50
            submit.draw()
            submit = arcade.Sprite("refresh.png",0.5)
            submit.center_x,submit.center_y = 560, 50
            submit.draw()
        else:
            # Text on the top
            arcade.draw_text("List of films that we recommend you:",
                            0,SCREEN_HEIGHT-40,
                            width=SCREEN_WIDTH,
                            font_size=24,
                            align="center")
            for i in range(len(self.results)):
                # Drawing titles of films that were 
                # recommended by the recommender 
                arcade.draw_text(f"{3-i}. {self.results[i]}",
                                0,SCREEN_HEIGHT/(USERS_NUM+5) * i + 300,
                                arcade.color.WHITE,24,align="center",width=SCREEN_WIDTH)

    def setup(self):
        """ Set up the window and initialize the variables. """
        self.buttons_list = None
        self.users_list = []
        self.previous = -1
        self.clicked = -1
        self.screen_num = 0
        self.results = []
        self.buttons_list = arcade.SpriteList()
        # Generating random users to display their films
        for i in range(USERS_NUM):
            self.users_list.append(str(r.randint(1,60)))
            # Buttons drawing
            button = arcade.Sprite('btn.png',1)
            button.center_x = SCREEN_WIDTH/2
            button.center_y = SCREEN_HEIGHT/(USERS_NUM+2) * i + 150
            self.buttons_list.append(button)

    def on_update(self,delta_time):
        """ Updates and window logic """
        # Changing texture of a button when user selects some option
        if(self.previous != -1):
            x,y = self.buttons_list[self.previous].center_x,self.buttons_list[self.previous].center_y
            self.buttons_list[self.previous] = arcade.Sprite('btn.png',1)
            self.buttons_list[self.previous].center_x = x
            self.buttons_list[self.previous].center_y = y
        if(self.clicked != -1):
            x,y = self.buttons_list[self.clicked].center_x,self.buttons_list[self.clicked].center_y
            self.buttons_list[self.clicked] = arcade.Sprite('btn_pressed.png',1)
            self.buttons_list[self.clicked].center_x = x
            self.buttons_list[self.clicked].center_y = y

    def on_mouse_press(self, x, y, button, modifiers):
        """ Mouse clicking handling """
        if(self.screen_num==0):
            # box 0 coords
            if(x>170 and x<633 and y<181 and y>115):
                self.previous = self.clicked
                self.clicked = 0
            # box 1 -||- 
            elif(x>170 and x<633 and y<272 and y>199):
                self.previous = self.clicked
                self.clicked = 1
            # box 2
            elif(x>170 and x<633 and y<354 and y>287):
                self.previous = self.clicked
                self.clicked = 2
            # box 3
            elif(x>170 and x<633 and y<439 and y>376):
                self.previous = self.clicked
                self.clicked = 3
            # box 4
            elif(x>170 and x<633 and y<523 and y>460):
                self.previous = self.clicked
                self.clicked = 4
            # Submit button
            elif(x>475 and x < 525 and y>25 and y < 75):
                self.screen_num=1
                x = predict_movie(self.users_list[self.clicked],3)
                for i in x:
                    self.results.append(str(i))
            # Refresh button
            elif(x>535 and x < 585 and y>25 and y < 75):
                for i in range(USERS_NUM):
                    self.users_list[i]=str(r.randint(1,60))
        else:
            # Reseting window after displaying recommendations
            self.setup()

def main():
    """ Main function """
    window = App()
    window.setup()
    arcade.run()

if __name__ == "__main__":
    main()