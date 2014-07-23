package com.micheledamian.signrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

class SignRecognitionView extends SignRecognitionViewBase {

	private static final String TAG = "SignRecognition::View";
	
	int frame = 0;
	
    public SignRecognitionView(Context context) {
        super(context);
        init();
        Log.i(TAG, "View created");
    }

    @Override
    protected Bitmap processFrame(byte[] data) {
        	
    	int H = getFrameHeight();
    	int W = getFrameWidth();
        int[] rgba = new int[H * W];
    
        Log.i(TAG, "Frame " + frame++);
    	
        findSign(W, H, data, rgba);

        Bitmap bmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888);
        bmp.setPixels(rgba, 0, W, 0, 0, W, H);
        
        return bmp;
    
    }

    public native void findSign(int width, int height, byte yuv[], int[] rgba);
    public native void init();

    static {
        System.loadLibrary("engine");
    }
}
