package com.micheledamian.signrecognition;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;

public class SignRecognitionActivity extends Activity {
	
	private static final String TAG = "SignRecognition::Activity";

    public SignRecognitionActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

	
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
    	super.onCreate(savedInstanceState);
    	Log.i(TAG, "onCreate");
    	
    	requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);        
        
        setContentView(new SignRecognitionView(this));
    
    }
}