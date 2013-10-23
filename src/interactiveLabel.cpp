/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "interactiveLabel.h"
#include "imageProcess.h"
#include "dataset.h"
#include "classes.h"
#include "class.h"
#include "pose.h"
#include "spatialModel.h"


InteractiveLabelingSession::InteractiveLabelingSession(ImageProcess *p, PartLocalizedStructuredLabelWithUserResponses *responses, bool isInteractive, 
						       bool drawPoints, bool drawLabels, bool drawRects, bool drawTree, 
						       float zoom, const char *debugDir, bool debugProbabilityMaps) {
  this->process = p;
  this->responses = responses;
  this->isInteractive = isInteractive;

  process->SetBidirectional(true);

  this->htmlDebugDir = debugDir ? StringCopy(debugDir) : NULL;
  this->debugProbabilityMaps = debugProbabilityMaps;

  dragPart = overPart = -1;
  numMoved = 0;
  movedPartSequence = NULL;
  partLocationHistory = NULL;
  partLocations = NULL;
  totalLosses = partDists = NULL;

  this->drawRects = drawRects;
  this->drawLabels = drawLabels;
  this->drawPoints = drawPoints;
  this->drawTree = drawTree;

  this->zoom = zoom;
}

InteractiveLabelingSession::~InteractiveLabelingSession() {
  if(htmlDebugDir) free(htmlDebugDir);
  if(movedPartSequence) {
    for(int i = 0; i < numMoved; i++)
      delete movedPartSequence[i];
    free(movedPartSequence);
  }
  if(partLocationHistory) {
    for(int i = 0; i <= numMoved; i++)
      delete [] partLocationHistory[i];
    free(partLocationHistory);
  }
  if(totalLosses) free(totalLosses);
  if(partDists) free(partDists);
}


void cvInteractiveMouseHandler(int event, int x, int y, int flags, void* param);


float InteractiveLabelingSession::GetLoss(PartLocation *locs, float *losses, int *worstPart, float stopThresh, float *totalLoss) {
  PartLocation *gt_locs = responses->GetPartLocations();
  float maxDist = 0;

  if(totalLoss) *totalLoss = 0;
  if(worstPart) *worstPart = -1;
  for(int j = 0; j < process->GetClasses()->NumParts(); j++) {
    float pred_x, pred_y, gt_x, gt_y;
    int pred_pose_ind, gt_pose_ind;
    locs[j].GetImageLocation(&pred_x, &pred_y);
    gt_locs[j].GetImageLocation(&gt_x, &gt_y);
    locs[j].GetDetectionLocation(NULL, NULL, NULL, NULL, &pred_pose_ind);
    gt_locs[j].GetDetectionLocation(NULL, NULL, NULL, NULL, &gt_pose_ind);
    int *num;
    ObjectPartPoseTransitionInstance ***t = process->GetClickPartInst(j)->GetVisiblePose()->GetPosePartTransitions(&num);
    ObjectPoseInstance *gt_pose = gt_pose_ind >= 0 ? process->GetPartInst(j)->GetPose(gt_pose_ind) : NULL;
    ObjectPoseInstance *pose = pred_pose_ind >= 0 ? process->GetPartInst(j)->GetPose(pred_pose_ind) : NULL;
    float wx, wy, wxx, wyy;
    t[0][0]->Model()->GetWeights(&wx, &wxx, &wy, &wyy);
    float dist = gt_pose->IsNotVisible() != pose->IsNotVisible() ? stopThresh+.00001 : 
      (gt_pose->IsNotVisible() ? 0 : sqrt((SQR(pred_x - gt_x) * wxx + SQR(pred_y - gt_y) * wyy)/2/process->GetClasses()->GetClickGamma()));
    losses[j] = dist;
    
    if(dist > maxDist) {
      maxDist = dist;
      if(worstPart) *worstPart = j;
    }
    if(totalLoss && dist > stopThresh)
      *totalLoss++;
  }
  return maxDist;
}


double InteractiveLabelingSession::Preprocess() {
  // Preprocess the image
  double score = process->Detect();

  elapsedTime = 0;
  partLocationHistory = (PartLocation**)realloc(partLocationHistory, sizeof(PartLocation*)*(numMoved+1));
  partLocations = partLocationHistory[0] = process->ExtractPartLocations();  

  if(htmlDebugDir)
    PrintDebugInfo();

  return score;
}

PartLocation *InteractiveLabelingSession::Label(int maxActions, float stopThresh) {
  Classes *classes = process->GetClasses();
  float losses[10000];

  Preprocess();

  if(isInteractive) {
    // Allow the user to interactively label the image
    IplImage *img = (process->Features()->GetImage());
    IplImage *img2 = cvCreateImage(cvSize(img->width*zoom,img->height*zoom), img->depth, img->nChannels);
    cvResize(img, img2);
    process->Draw(img2, partLocations, CV_RGB(0,0,255), drawLabels, false, drawPoints, drawRects, true, dragPart, drawTree);
    cvNamedWindow("win1", CV_WINDOW_AUTOSIZE); 
    cvShowImage("win1", img2);
    cvReleaseImage(&img2);
    cvSetMouseCallback("win1", cvInteractiveMouseHandler, this);
    cvWaitKey(0);
  } else {
    assert(responses);

    PartLocation *gt_locs = responses->GetPartLocations();
    int worstPart=-1;
    float totalLoss;
    for(int i = 0; i < maxActions && i <= classes->NumParts(); i++) {
      // Simulate user: find the part with the greatest localization error, or stop when all parts are close enough
      float maxDist = GetLoss(partLocations, losses, &worstPart, stopThresh, &totalLoss);
      if(maxDist <= stopThresh)
	break;
      PartLocation l(gt_locs[worstPart]);
      FinalizePartLocation(&l, maxDist, totalLoss);
    } 
  }

  if(htmlDebugDir) {
    char html_name[1000];
    sprintf(html_name, "%s/%s.html", htmlDebugDir, process->Features()->Name());
    FILE *fout = fopen(html_name, "a");
    fprintf(fout, "</table></body></html>");
    fclose(fout);
  }

  return partLocations;
}

void InteractiveLabelingSession::FinalizePartLocation(PartLocation *loc, float partDist, float totalLoss) {
  // When the user releases the mouse after dragging a part, map the change into
  // a unary potential for that part then propagate it to the other parts
  ObjectPartInstance *part = process->GetPartInst(loc->GetPartID());
  part->SetLocation(loc, false);
  part->PropagateUnaryMapChanges();


  partDists = (float*)realloc(partDists, sizeof(float)*(numMoved+1));
  partDists[numMoved] = partDist;
  totalLosses = (float*)realloc(totalLosses, sizeof(float)*(numMoved+1));
  totalLosses[numMoved] = totalLoss;

  movedPartSequence = (PartLocation**)realloc(movedPartSequence, sizeof(PartLocation*)*(numMoved+1));
  movedPartSequence[numMoved] = new PartLocation(*loc);

  numMoved++;
  partLocationHistory = (PartLocation**)realloc(partLocationHistory, sizeof(PartLocation*)*(numMoved+1));
  PartLocation *locs = PartLocation::NewPartLocations(process->GetClasses(), process->Image()->width, process->Image()->height, process->Features(), false);
  part->ExtractPartLocations(locs);
  partLocations = partLocationHistory[numMoved] = locs;//process->ExtractPartLocations();
  //assert(loc->x == partLocations[part->Id()].x && loc->y == partLocations[part->Id()].y);


  if(htmlDebugDir)
    PrintDebugInfo();
}

int InteractiveLabelingSession::PointOnPart(int x, int y) {
  assert(partLocations);
  for(int i = process->GetClasses()->NumParts()-1; i >= 0; i--) {
    ObjectPart *part = process->GetClasses()->GetPart(i);
    float px, py;
    int pose;
    partLocations[i].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
    partLocations[i].GetImageLocation(&px, &py);
    if((pose < 0 || !part->GetPose(pose)->IsNotVisible()) && SQR(x-px*zoom)+SQR(y-py*zoom) <= SQR(CIRCLE_RADIUS))
      return i;
  }
  return -1;
}

void InteractiveLabelingSession::PrintDebugInfo() {
  char fname[1000], html_name[1000], *html = (char*)malloc(200000);
  
  sprintf(html_name, "%s/%s.html", htmlDebugDir, process->Features()->Name());
  FILE *fout;
  if(!numMoved) { 
    char fname[1000]; strcpy(fname, process->ImageName());
    sprintf(fname, "%s/%s_gt.png", htmlDebugDir, process->Features()->Name());
    if(responses) {
      IplImage *img = cvCloneImage(process->Features()->GetImage());
      process->Draw(img, responses->GetPartLocations());
      cvSaveImage(fname, img);
      cvReleaseImage(&img);
    } else {
      cvSaveImage(fname, process->Features()->GetImage());
    }
    sprintf(fname, "%s_gt.png", process->Features()->Name());
      
    
    fout = fopen(html_name, "w"); 
    fprintf(fout, "<html><body><h1>%s</h1>\n<img src=\"%s\"><br><br>\n", process->Features()->Name(), fname);
    fprintf(fout, "\n\n<br><table>");
  } else {
    fout = fopen(html_name, "a");
  }
  assert(fout);

  IplImage *img = cvCloneImage(process->Features()->GetImage());
  process->Draw(img, partLocations, CV_RGB(0,0,255), drawLabels, false, drawPoints, drawRects, true,  -1, drawTree);
  if(numMoved) {
    int id = movedPartSequence[numMoved-1]->GetPartID();
    float x1, y1, x2, y2;
    partLocationHistory[numMoved-1][id].GetImageLocation(&x1, &y1);
    partLocationHistory[numMoved][id].GetImageLocation(&x2, &y2);
    cvLine(img, cvPoint(x1,y1), cvPoint(x2,y2), CV_RGB(0,255,0), 3);
  }
  sprintf(fname, "%s/%s_q%d.png", htmlDebugDir, process->Features()->Name(), numMoved);
  cvSaveImage(fname, img);
  cvReleaseImage(&img);
  sprintf(fname, "%s_q%d.png", process->Features()->Name(), numMoved);
  fprintf(fout, "\n<tr><td><img src=\"%s\">", fname);
  if(numMoved) fprintf(fout, "<br>%s=%f, %f", process->GetClasses()->GetPart(movedPartSequence[numMoved-1]->GetPartID())->Name(), 
		       partDists[numMoved-1], totalLosses[numMoved-1]);
  

  int x1, y1, x, y;
  PartLocation *gt_locs = responses ? responses->GetPartLocations() : NULL;
  if(gt_locs) {
  int worstPart;
  float losses[10000], totalLoss;
  GetLoss(partLocations, losses, &worstPart, 1.5, &totalLoss);  
  for(int i = 0; i < process->GetClasses()->NumParts(); i++) {
    gt_locs[i].GetDetectionLocation(&x1, &y1);
    partLocations[i].GetDetectionLocation(&x, &y);
    fprintf(fout, "<br>%s: loss=%f gt=(%d,%d) pred=(%d,%d)\n", process->GetClasses()->GetPart(i)->Name(),
	    losses[i], x1, y1, x, y);
  }
  }


  fprintf(fout, "</td>\n");
  if(debugProbabilityMaps) {
    strcpy(html, "");
    sprintf(fname, "%s_%d", process->Features()->Name(), numMoved);
    process->SaveProbabilityMaps(fname, htmlDebugDir, html);
    fprintf(fout, "%s\n", html);
  }
  fprintf(fout, "</tr>\n<tr><td colspan=%d></td></tr>\n", process->GetClasses()->NumPoses());

  fclose(fout);
  free(html);
}


void cvInteractiveMouseHandler(int event, int x, int y, int flags, void* param) {
  InteractiveLabelingSession *s = (InteractiveLabelingSession*)param;
  ImageProcess *process = s->Process();
  FeatureOptions *feat = process->Features();
  IplImage *img = feat->GetImage();
  CvPoint dragPt;
  int dragPart = s->DragPart(&dragPt);
  int overPart = s->OverPart();
  int dispPart = dragPart;
  PartLocation *locs, l_tmp;
  bool draw = dragPart >= 0;
  float zoom = s->Zoom();

  if(dragPart >= 0) {
    l_tmp.Init(process->GetClasses(), process->Image()->width, process->Image()->height, feat);
    l_tmp.SetPart(process->GetClasses()->GetPart(dragPart));
    l_tmp.SetImageLocation((float)x/zoom, (float)y/zoom, LATENT, LATENT, NULL);
  }

  switch(event){
  case CV_EVENT_MOUSEMOVE:
    if(dragPart < 0) {
      int o = s->PointOnPart(x, y);
      if(o != overPart) {
	s->SetOverPart(o);
	dispPart = o;
	draw = true;
      }
    }
    if(draw) {
      locs = PartLocation::NewPartLocations(process->GetClasses(), process->Image()->width, process->Image()->height, feat, false);
      if(dragPart >= 0) {
	locs[dragPart] = l_tmp;
	process->GetPartInst(dispPart)->ExtractPartLocations(locs, &l_tmp, NULL);
      } else {
	PartLocation *l = s->GetPartLocations();
	for(int i = 0; i < process->GetClasses()->NumParts(); i++)
	  locs[i] = l[i];
      }
      IplImage *img2 = cvCreateImage(cvSize(img->width*zoom,img->height*zoom), img->depth, img->nChannels);
      cvResize(img, img2);
      process->Draw(img2, locs, CV_RGB(0,0,255), s->DrawLabels(), false, s->DrawPoints(), s->DrawRects(), true, dispPart, s->DrawTree());
      if(dragPart >= 0)
	cvLine(img2, cvPoint(dragPt.x,dragPt.y), cvPoint(x,y), CV_RGB(0,255,0), 3);
      cvShowImage("win1", img2);
      cvReleaseImage(&img2);
      delete [] locs;
    }   
    break;
  case CV_EVENT_LBUTTONDOWN:
    s->SetDragPart(s->OverPart(), cvPoint(x,y));
    break;
  case CV_EVENT_LBUTTONUP:
    if(dragPart >= 0)
      s->FinalizePartLocation(&l_tmp);
    s->SetDragPart(-1, cvPoint(0,0));
    break;
  }
}
