from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.app import App
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast

 
# For mobile phone
#from android.storage import primary_external_storage_path
#primary_ext_storage = primary_external_storage_path()
 
class ProfilePhotoManager(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            #preview=True
        )
 
 
    def file_manager_open(self):
        self.file_manager.show('/')  # for computer
        #self.file_manager.show(primary_ext_storage)  # for mobile phone
        self.manager_open = True
 
    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.
 
        :type path: str;
        :param path: path to the selected directory or file;
        '''
 
        self.exit_manager()
        toast(path)
        App.get_running_app().change_profile_source(path)
        print(path)
 
    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''
 
        self.manager_open = False
        self.file_manager.close()
 
    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''
 
        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True