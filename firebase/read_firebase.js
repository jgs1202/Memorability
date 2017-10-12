firebase ref = new firebase("https://memorability-a0c21.firebaseio.com/message");

ref.addValueEventListener(new ValueEventListener() {
    @Override
    public void onDataChange(DataSnapshot snapshot) {
        for (DataSnapshot dataSnapshot : snapshot.getChildren()) {
            String sender = (String) dataSnapshot.child("sender").getValue();
            String body = (String) dataSnapshot.child("body").getValue();
            Log.d("Firebase", String.format("sender:%s, body:%s", sender, body));
        }
    }

    @Override
    public void onCancelled(FirebaseError error) {
    }
});
